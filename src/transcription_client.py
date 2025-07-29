
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import deque

from fastapi import WebSocket
from faster_whisper import WhisperModel

from models.messages import ErrorMessage, TranscriptionSegmentMessage
from utils import wrap_pcm_as_wav

WORD_OVERLAP_MARGIN_SECONDS = 0.1

class SlidingWindowAudioBuffer:
    def __init__(self, capacity: int, emit_threshold_bytes : int):
        self.capacity = capacity
        self.queue = deque()
        self.lock = asyncio.Lock()
        self.stopped = False
        self.buffer_size = 0
        self.total_size = 0
        self.emit_threshold_bytes = emit_threshold_bytes
        self.accum_since_last_yield = 0

    async def append(self, data: bytes):
        async with self.lock:
            self.queue.append(data)
            self.buffer_size += len(data)
            self.total_size += len(data)
            self.accum_since_last_yield += len(data)

    def stop(self):
        self.stopped = True

    def is_drained(self):
        return self.stopped and self.buffer_size == 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            await asyncio.sleep(0.01)
            async with self.lock:
                if self.accum_since_last_yield >= self.emit_threshold_bytes:
                    chunk = b''.join(self.queue)
                    self.accum_since_last_yield = 0
                    while self.buffer_size > self.capacity:
                        tail = self.queue.popleft()
                        tail_size = len(tail)
                        self.buffer_size -= tail_size

                    return chunk

                elif self.stopped:
                    if self.buffer_size > 0:
                        chunk = b''.join(self.queue)
                        self.queue.clear()
                        self.buffer_size = 0
                        return chunk
                    raise StopAsyncIteration
                

class TranscriptionClient():

    def __init__(self, 
        executor: ThreadPoolExecutor,
        model: WhisperModel, 
        ws: WebSocket, 
        language = "en", 
        transmit_interval_seconds: float = 2.0,
        context_delay_seconds: float = 1.0,
        audio_lookback_seconds: float = 4.0,
        sample_rate = 16000,
        sample_width = 2,
        channels = 1
    ):
        self.language = language
        self.worker: asyncio.Task = None
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels

        buffer_capacity_bytes = audio_lookback_seconds * sample_rate * sample_width * self.channels
        self.context_delay_seconds = context_delay_seconds

        emit_threshold_bytes = (transmit_interval_seconds - context_delay_seconds) * sample_rate * sample_width * self.channels
        self.audiobuffer = SlidingWindowAudioBuffer(capacity=buffer_capacity_bytes, emit_threshold_bytes=emit_threshold_bytes)
        self.executor = executor
        self.model = model
        self.ws = ws
        self.latest_sent_segment_end_timestamp: float = 0
        self.active = False

    async def start(self) -> None:
        self.worker = asyncio.create_task(self.transcription_task())
        self.active = True

    async def end(self):
        self.audiobuffer.stop()
        if self.worker and not self.worker.done():
            await self.worker
        self.active = False

    async def cancel(self):
        if self.worker and not self.worker.done():
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                print("Transcription cancelled succesfully")
            except Exception as e:
                print(f"Unexpected error while cancelling transcription: {e}")
        self.active = False

    async def transcription_task(self):
        loop = asyncio.get_running_loop()
        async for chunk in self.audiobuffer:
            if not chunk:
                continue

            audio_stream = wrap_pcm_as_wav(
                chunk,
                sample_rate=self.sample_rate,
                sample_width=self.sample_width,
                channels=self.channels
            )
            buffer_tail_ts = (self.audiobuffer.total_size - len(chunk)) / (self.sample_rate * self.sample_width * self.channels)

            buffer_cuttoff_ts = (self.audiobuffer.total_size) / (self.sample_rate * self.sample_width * self.channels) - self.context_delay_seconds

            try:
                segments, _ = await loop.run_in_executor(
                    self.executor,
                    lambda: self.model.transcribe(
                        audio_stream,
                        language=self.language,
                        word_timestamps=True
                    )
                )

                for seg in segments:
                    seg_start_ts = seg.start + buffer_tail_ts
                    seg_end_ts = seg.end + buffer_tail_ts

                    if seg_end_ts < self.latest_sent_segment_end_timestamp:
                        continue
                    else:
                        text = ""
                        for word in seg.words:
                            word_start_ts = word.start + buffer_tail_ts 
                            word_end_ts = word.end + buffer_tail_ts

                            if word_start_ts > buffer_cuttoff_ts and not self.audiobuffer.is_drained():
                                break

                            if word_start_ts + WORD_OVERLAP_MARGIN_SECONDS < self.latest_sent_segment_end_timestamp:
                                continue

                            text += word.word
                            self.latest_sent_segment_end_timestamp = word_end_ts

                        else:
                            if text:
                                msg = TranscriptionSegmentMessage(
                                    text=text,
                                    start=word_start_ts,
                                    end=word_end_ts,
                                )
                                await self.ws.send_text(msg.json())
                            continue

                        if text:
                            msg = TranscriptionSegmentMessage(
                                text=text,
                                start=word_start_ts,
                                end=word_end_ts,
                            )
                            await self.ws.send_text(msg.json())
                        break

            except Exception as e:
                msg = ErrorMessage(detail=str(e))
                await self.ws.send_text(msg.json())

    async def append_chunk(self, audio_chunk: bytes):
        await self.audiobuffer.append(audio_chunk)


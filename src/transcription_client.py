
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from collections import deque

from fastapi import WebSocket
from faster_whisper import WhisperModel

from src.models.messages import ErrorMessage, TranscriptionSegmentMessage
from src.utils import wrap_pcm_as_wav

class SlidingWindowAudioBuffer:
    def __init__(self, max_size: int, chunk_size: int):
        self.max_size = max_size
        self.queue = deque()
        self.lock = asyncio.Lock()
        self.stopped = False
        self.buffer_size = 0
        self.total_size = 0
        self.accum_since_last_yield_threshold = chunk_size
        self.accum_since_last_yield = 0

    async def append(self, data: bytes):
        async with self.lock:
            self.queue.append(data)
            self.buffer_size += len(data)
            self.total_size += len(data)
            self.accum_since_last_yield += len(data)

    def stop(self):
        self.stopped = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            await asyncio.sleep(0.01)
            async with self.lock:
                if self.accum_since_last_yield >= self.accum_since_last_yield_threshold:
                    chunk = b''.join(self.queue)
                    self.accum_since_last_yield = 0
                    while self.buffer_size > self.max_size:
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
        transcript_interval_seconds: float = 1.0,
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

        self.chunk_size = transcript_interval_seconds * sample_rate * sample_width * self.channels
        buffer_max_size = audio_lookback_seconds * sample_rate * sample_width * self.channels

        self.audiobuffer = SlidingWindowAudioBuffer(max_size=buffer_max_size, chunk_size=self.chunk_size)
        self.stop_flag = False
        self.executor = executor
        self.model = model
        self.ws = ws
        self.latest_segment_end_timestamp: float = 0
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
            buffer_tail_timestamp = (self.audiobuffer.total_size - len(chunk)) / (self.sample_rate * self.sample_width * self.channels)

            try:
                segments, _ = await loop.run_in_executor(
                    self.executor,
                    lambda: self.model.transcribe(
                        audio_stream,
                        language=self.language,
                        word_timestamps=True
                    )
                )

                dedup_needed = True    

                for seg in segments:
                    seg_start_ts = seg.start + buffer_tail_timestamp
                    seg_end_ts = seg.end + buffer_tail_timestamp

                    if not dedup_needed:
                        msg = TranscriptionSegmentMessage(
                            text=seg.text,
                            start=seg_start_ts,
                            end=seg_end_ts,
                        )
                        await self.ws.send_text(msg.json())
                        self.latest_segment_end_timestamp = seg_end_ts
                    elif seg_end_ts < self.latest_segment_end_timestamp:
                        continue
                    else:
                        deduped_text = ""
                        for word in seg.words:
                            if not dedup_needed:
                                deduped_text += word.word
                                continue

                            word_start_ts = word.start + buffer_tail_timestamp

                            if word_start_ts < self.latest_segment_end_timestamp:
                                continue

                            deduped_text += word.word
                            dedup_needed = False

                        msg = TranscriptionSegmentMessage(
                            text=deduped_text,
                            start=seg_start_ts,
                            end=seg_end_ts,
                        )
                        await self.ws.send_text(msg.json())
                        self.latest_segment_end_timestamp = seg_end_ts

            except Exception as e:
                msg = ErrorMessage(detail=str(e))
                await self.ws.send_text(msg.json())

    async def append_chunk(self, audio_chunk: bytes):
        await self.audiobuffer.append(audio_chunk)


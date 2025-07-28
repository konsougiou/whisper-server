import io
import wave

def wrap_pcm_as_wav(pcm_bytes: bytes, sample_rate=16000, channels=1, sample_width=2) -> io.BytesIO:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buffer.seek(0)
    return buffer
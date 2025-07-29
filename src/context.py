from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
import os

MAX_CONNECTIONS = int(os.getenv("MAX_WS_CONNECTIONS", 4))

executor = ThreadPoolExecutor(max_workers=MAX_CONNECTIONS)

whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
from pydantic import BaseModel
from typing import Literal

from models.enums import DisconnectReason

class BaseMessage(BaseModel):
    type: str

class OpenParameters(BaseModel):
    transmit_interval_seconds: float
    context_delay_seconds: float
    audio_lookback_seconds: float
    language: str
    sample_rate: int
    sample_width: int
    channels: int

class OpenMessage(BaseMessage):
    type: Literal["open"] = "open"
    parameters: OpenParameters

class OpenedMessage(BaseMessage):
    type: Literal["opened"] = "opened"


class CloseMessage(BaseMessage):
    type: Literal["close"] = "close"
    reason: str

class ClosingMessage(BaseMessage):
    type: Literal["closing"] = "closing"

class ClosedMessage(BaseMessage):
    type: Literal["closed"] = "closed"

class TranscriptionSegmentMessage(BaseMessage):
    type: Literal["transcription_segment"] = "transcription_segment"
    text: str
    start: float
    end: float


class ErrorMessage(BaseMessage):
    type: Literal["error"] = "error"
    detail: str

class DisconnectMessage(BaseMessage):
    type: Literal["disconnect"] = "disconnect"
    reason: DisconnectReason
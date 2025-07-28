from uuid import UUID, uuid4
from enum import StrEnum

from pydantic import BaseModel, Field
from src.transcription_client import TranscriptionClient


class SessionState(StrEnum):
    PENDING = "pending"
    OPENED = "opened"
    CLOSED = "closed"
    CLOSING = "closing"


class Session:
    id: UUID = uuid4()
    state: SessionState = SessionState.PENDING
    transcription_client: TranscriptionClient | None = None
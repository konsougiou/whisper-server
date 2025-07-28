from enum import StrEnum


class ServerMessageType(StrEnum):
    OPENED = "opened"
    CLOSED = "closed"
    CLOSING = "closing"
    ERROR = "error"
    TRANSCRIPTION_SEGMENT = "transcription_segment"
    DISCONNECT = "disconnect"


class ClientMessageType(StrEnum):
    OPEN = "open"
    CLOSE = "close"
    AUDIO = "audio"
    DISCONNECT = "disconnect"


class DisconnectReason(StrEnum):
    ERROR = "error"
    COMPLETED = "completed"




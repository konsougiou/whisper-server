from fastapi import WebSocket

from models.messages import CloseMessage, ClosedMessage, ClosingMessage, OpenMessage, ErrorMessage, OpenedMessage
from models.session import Session, SessionState
from transcription_client import TranscriptionClient

from context import executor, whisper_model

async def handle_open_message(session: Session, ws: WebSocket, message: OpenMessage):

    if session.state == SessionState.OPENED:
        detail = "Cannot open already opened session"
        server_msg = ErrorMessage(detail=detail)
        await ws.send_text(server_msg.json()) 

    elif session.state == SessionState.CLOSED:
        detail = "Cannot reopen a closed session"
        server_msg = ErrorMessage(detail=detail)
        await ws.send_text(server_msg.json())  

    else:
        client_kwargs = message.parameters.dict()
        session.transcription_client = TranscriptionClient(executor, whisper_model, ws, **client_kwargs)
        await session.transcription_client.start() 
        server_msg = OpenedMessage()
        await ws.send_text(server_msg.json())


async def handle_close_message(session: Session, ws: WebSocket, message: CloseMessage):
    # TODO: Logging
    if session.transcription_client.active:
        session.state = SessionState.CLOSING
        server_msg = ClosingMessage()
        await ws.send_text(server_msg.json())

        await session.transcription_client.end()

        session.state = SessionState.CLOSED
        server_msg = ClosedMessage()
        await ws.send_text(server_msg.json())
    

async def handle_disconnect_message(session: Session, ws, message):
    if session.transcription_client.active:
        await session.transcription_client.cancel()


async def handle_bytes(session: Session, ws: WebSocket, data: bytes):
    if session.transcription_client.active:
        await session.transcription_client.append_chunk(data)
    else:
        await ws.send_text(ErrorMessage(detail = "Received audio before stream start."))


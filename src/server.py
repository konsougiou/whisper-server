from concurrent.futures import ThreadPoolExecutor
import json
import os
from asyncio import Semaphore
from typing import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from pydantic import ValidationError

from src.models.session import Session
from src.models.messages import (
    ErrorMessage,
    OpenMessage,
    CloseMessage,
    DisconnectMessage,
    BaseMessage,
)
from src.models.enums import ClientMessageType
from src.message_handlers import (
    handle_open_message,
    handle_close_message,
    handle_bytes,
    handle_disconnect_message,
)
from src.context import MAX_CONNECTIONS

connection_semaphore = Semaphore(MAX_CONNECTIONS)

app = FastAPI()

MESSAGE_HANDLERS: dict[ClientMessageType, tuple[type[BaseMessage], Callable]] = {
    ClientMessageType.OPEN: (OpenMessage, handle_open_message),
    ClientMessageType.CLOSE: (CloseMessage, handle_close_message),
    ClientMessageType.DISCONNECT: (DisconnectMessage, handle_disconnect_message),
}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    if not connection_semaphore.locked():
        await ws.accept()
    else:
        await ws.close(code=1013)
        return

    session = Session()

    try:
        async with connection_semaphore:
            while True:
                message = await ws.receive()

                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        await handle_bytes(session, ws, message["bytes"])

                    elif "text" in message:
                        try:
                            base = BaseMessage.parse_raw(message["text"])
                            try:
                                msg_type = ClientMessageType(base.type)
                            except ValueError:
                                await ws.send_text(ErrorMessage(detail = f"Unknown message type: {base.type}"))
                                continue

                            if msg_type in MESSAGE_HANDLERS:
                                ModelClass, handler = MESSAGE_HANDLERS[msg_type]
                                parsed_msg = ModelClass.parse_raw(message["text"])
                                await handler(session, ws, parsed_msg)
                            else:
                                await ws.send_text(ErrorMessage(detail = f"Unknown message type: {base.type}"))
                                

                        except ValidationError as e:
                            await ws.send_text(ErrorMessage(detail = f"Invalid message format: {e.errors()}"))
                        except json.JSONDecodeError:
                            await ws.send_text(ErrorMessage(detail = "Invalid JSON"))

                elif message["type"] == "websocket.disconnect":
                    if session.transcription_client and session.transcription_client.active:
                        await session.transcription_client.cancel()
                    break

    except WebSocketDisconnect:
        if session.transcription_client and session.transcription_client.active:
            await session.transcription_client.cancel()

from typing import Dict
from fastapi import WebSocket

from app.core.logger import get_logger
from app.services.streaming.stream_session_service import StreamSession

logger = get_logger(__name__)


class ConnectionManager:
    def __init__(self):
        # session_id -> websocket
        self.active_connections: Dict[str, WebSocket] = {}

        # session_id -> StreamSession
        self.sessions: Dict[str, StreamSession] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        self.active_connections[session_id] = websocket
        logger.info(f"Client connected | session_id={session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

        if session_id in self.sessions:
            self.sessions[session_id].close()
            del self.sessions[session_id]

        logger.info(f"Client disconnected | session_id={session_id}")

    def create_session(self, session_id: str) -> StreamSession:
        session = StreamSession(session_id=session_id)
        self.sessions[session_id] = session
        logger.info(f"Session created | session_id={session_id}")
        return session

    def get_session(self, session_id: str) -> StreamSession:
        return self.sessions.get(session_id)

    async def send_json(self, session_id: str, data: dict):
        websocket = self.active_connections.get(session_id)

        if websocket:
            await websocket.send_json(data)
        else:
            logger.warning(f"Send failed — no websocket | session_id={session_id}")

    async def broadcast(self, data: dict):
        for session_id, websocket in self.active_connections.items():
            await websocket.send_json(data)
            logger.debug(f"Broadcast | session_id={session_id} | type={data.get('type')}")

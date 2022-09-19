from collections import defaultdict
from typing import List

from databases import Database
from fastapi import WebSocket
from pydantic import BaseSettings


# https://fastapi.tiangolo.com/advanced/settings/
class Settings(BaseSettings):
    app_name: str = "SketchAPI"
    db_url: str = "sqlite+aiosqlite:///test.db"
    debug: bool = False
    faiss_path: str = "."
    setup_fake_users: bool = True


settings = Settings()

database = Database(
    settings.db_url,
)

local_cache = {}


class SocketManager:
    def __init__(self):
        self.threads = defaultdict(list)

    async def connect(self, thread_id, websocket: WebSocket, user: str):
        await websocket.accept()
        self.threads[thread_id].append((websocket, user))

    def disconnect(self, thread_id, websocket: WebSocket, user: str):
        self.threads[thread_id].remove((websocket, user))

    async def broadcast(self, thread_id, data):
        for websocket, _ in self.threads[thread_id]:
            await websocket.send_json(data)


manager = SocketManager()

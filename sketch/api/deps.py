from databases import Database
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

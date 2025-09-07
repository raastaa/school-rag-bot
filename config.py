from functools import lru_cache
import logging
from logging.handlers import RotatingFileHandler
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BOT_TOKEN: str
    ADMIN_IDS: str = ""
    QDRANT_COLLECTION: str = "school_docs"
    TEACH_DIR: str = "./teach"
    QDRANT_PATH: str = "./qdrant_storage"

    GIGACHAT_CREDENTIALS: str = ""
    GIGACHAT_SCOPE: str = "GIGACHAT_API_PERS"
    GIGACHAT_EMBEDDINGS_MODEL: str = "Embeddings"
    GIGACHAT_BASE_URL: str = ""
    GIGACHAT_VERIFY_SSL: str = "true"
    GIGACHAT_CA_BUNDLE: str | None = None

    GOOGLE_API_KEY: str | None = None
    GOOGLE_CSE_ID_SMP: str | None = None
    GOOGLE_CSE_ID_WEB: str | None = None

    RELEVANCE_THRESHOLD: float = 0.82
    EMBEDDING_BATCH: int = 64

    APP_DB_PATH: str = "./app.db"

    EMBEDDING_MAX_TOKENS: int = 514
    EMBEDDING_TARGET_TOKENS: int = 480
    EMBEDDING_OVERLAP_TOKENS: int = 60

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "bot.log"

    WEB_SEARCH_CACHE_TTL: int = 300

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()


def setup_logging() -> None:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    if settings.LOG_FILE:
        handlers.append(RotatingFileHandler(settings.LOG_FILE, maxBytes=1_000_000, backupCount=3))
    logging.basicConfig(level=level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=handlers)

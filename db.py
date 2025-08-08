# db.py
from functools import lru_cache
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Подхват .env, даже если запуск не из корня
load_dotenv()

class Settings(BaseSettings):
    PG_DSN: str
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()  # валидирует наличие PG_DSN

@lru_cache
def get_engine():
    # создаём engine лениво, только при первом обращении
    return create_engine(get_settings().PG_DSN, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    enable_scheduler: bool = True
    scheduler_interval_hours: int = 24


@lru_cache
def get_settings() -> Settings:
    return Settings()

import re
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    enable_scheduler: bool = True
    scheduler_interval_hours: int = 24

    @field_validator("database_url", mode="after")
    @classmethod
    def normalize_database_url(cls, v: str) -> str:
        # Ensure asyncpg driver scheme
        if v.startswith("postgresql://") and "+asyncpg" not in v:
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        # asyncpg uses ssl= not sslmode=
        v = v.replace("sslmode=require", "ssl=require")
        # asyncpg doesn't support channel_binding — strip it
        v = re.sub(r"[?&]channel_binding=[^&]*", "", v)
        # Clean up any dangling ? or &
        v = re.sub(r"\?&", "?", v)
        v = re.sub(r"[?&]$", "", v)
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()

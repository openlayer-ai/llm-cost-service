from functools import lru_cache
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

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

        # Fix query parameters — parse properly to avoid mangling the URL
        parsed = urlparse(v)
        params = parse_qs(parsed.query, keep_blank_values=True)

        params.pop("channel_binding", None)  # not supported by asyncpg
        params.pop("sslmode", None)          # asyncpg uses ssl=, not sslmode=
        params.setdefault("ssl", ["require"])

        new_query = urlencode({k: vs[0] for k, vs in params.items()})
        return urlunparse(parsed._replace(query=new_query))


@lru_cache
def get_settings() -> Settings:
    return Settings()

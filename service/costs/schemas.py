from datetime import datetime

from pydantic import BaseModel, ConfigDict


class LlmCostSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    provider: str
    model: str
    prompt_cost_per_token: float
    completion_cost_per_token: float
    price_details: dict[str, float] = {}
    is_chat_capable: bool | None = None
    source: str
    updated_at: datetime


class ListCostsResponse(BaseModel):
    costs: list[LlmCostSchema]


class RefreshResponse(BaseModel):
    rows_affected: int
    rows_deleted: int = 0
    duration_ms: float
    per_source: dict[str, int] = {}
    failed_sources: list[str] = []


class StatusResponse(BaseModel):
    total_models: int
    total_providers: int
    last_refreshed_at: datetime | None

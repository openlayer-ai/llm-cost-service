from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class LlmCostEntity:
    """Per-token cost of an LLM model from a given provider.

    Attributes:
        provider: Lowercase provider name (e.g. "openai", "anthropic").
        model: Model identifier as used in API calls (e.g. "gpt-4o").
        prompt_cost_per_token: Cost in USD per prompt token.
        completion_cost_per_token: Cost in USD per completion token.
        source: Origin of this record (e.g. "litellm").
        updated_at: Timestamp of last DB update. None when built from a provider
                    before persistence.
    """

    provider: str
    model: str
    prompt_cost_per_token: float
    completion_cost_per_token: float
    source: str
    updated_at: datetime | None = field(default=None)

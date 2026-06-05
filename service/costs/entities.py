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
        price_details: Open-ended ``{token_category: cost_per_token}`` map for
                    granular per-category pricing (e.g. cached_tokens,
                    cache_creation_tokens, audio_input_tokens). Empty when the
                    source exposes no per-category prices. Consumers match these
                    keys to the same-named keys in a request's usageDetails.
        source: Origin of this record (e.g. "litellm").
        updated_at: Timestamp of last DB update. None when built from a provider
                    before persistence.
    """

    provider: str
    model: str
    prompt_cost_per_token: float
    completion_cost_per_token: float
    source: str
    price_details: dict[str, float] = field(default_factory=dict)
    updated_at: datetime | None = field(default=None)

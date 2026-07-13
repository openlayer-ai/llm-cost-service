from typing import Protocol

from service.costs.entities import LlmCostEntity

# Canonical per-category price keys — a contract shared with the SDK's
# ``usageDetails`` and the platform cost engine (keys are matched by name).
# This map holds only the GRANULAR EXTRAS: base input/output prices already live
# at the root as ``prompt_cost_per_token`` / ``completion_cost_per_token`` (the
# platform reuses those as ``input_tokens`` / ``output_tokens``), so they are not
# duplicated here. Reasoning is intentionally omitted too: it is billed at the
# output rate and already counted within output, so it carries no separate price
# and is surfaced only as an informational usage category.
_LITELLM_PRICE_FIELDS: dict[str, str] = {
    "cached_tokens": "cache_read_input_token_cost",
    "cache_creation_tokens": "cache_creation_input_token_cost",
    "audio_input_tokens": "input_cost_per_audio_token",
    "audio_output_tokens": "output_cost_per_audio_token",
}

# OpenRouter exposes per-category prices inside each model's ``pricing`` dict.
_OPENROUTER_PRICE_FIELDS: dict[str, str] = {
    "cached_tokens": "input_cache_read",
    "cache_creation_tokens": "input_cache_write",
}


def _coerce_price(value: object) -> float | None:
    """Return a non-negative float price, or None when not a usable number.

    LiteLLM gives floats; OpenRouter gives strings. ``bool`` is rejected (it is
    an ``int`` subclass).
    """
    if isinstance(value, bool):
        return None
    try:
        price = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return price if price >= 0 else None


def _build_price_details(
    source: dict, field_map: dict[str, str]
) -> dict[str, float]:
    """Map a provider entry's per-category cost fields to canonical price keys."""
    details: dict[str, float] = {}
    for canonical, field in field_map.items():
        price = _coerce_price(source.get(field))
        if price is not None:
            details[canonical] = price
    return details


# LiteLLM ``mode`` values that denote a model usable as a chat target. "chat"
# is the obvious one; "responses" covers Responses-API-only models (o1-pro,
# gpt-5-pro, gpt-5-codex, ...) — they serve /v1/responses rather than
# /v1/chat/completions but are still chat targets and must NOT be filtered out.
_LITELLM_CHAT_MODES: frozenset[str] = frozenset({"chat", "responses"})


def _litellm_is_chat_capable(entry: dict) -> bool | None:
    """Whether a LiteLLM model is a usable chat target, from its ``mode``.

    Non-chat modes (embedding, image_generation, audio_speech / transcription,
    moderation, rerank, ocr, video_generation, completion, ...) are not chat
    targets. Returns ``None`` when the entry carries no usable ``mode`` so the
    column stays NULL and consumers fall back to their own heuristic.
    """
    mode = entry.get("mode")
    if not isinstance(mode, str) or not mode:
        return None
    return mode.strip().lower() in _LITELLM_CHAT_MODES


def _openrouter_is_chat_capable(entry: dict) -> bool | None:
    """Whether an OpenRouter model is a usable chat target.

    OpenRouter is a chat/completion router, so capability is determined by
    whether the model can emit text — ``architecture.output_modalities``
    contains "text". Returns ``None`` when the architecture/modalities are
    absent so the column stays NULL (consumers fall back to their heuristic).
    """
    architecture = entry.get("architecture") or {}
    output_modalities = architecture.get("output_modalities")
    if not isinstance(output_modalities, list) or not output_modalities:
        return None
    return any(isinstance(m, str) and m.strip().lower() == "text" for m in output_modalities)


class CostDataProvider(Protocol):
    """Interface for fetching LLM cost data from an external source."""

    def fetch_costs(self) -> list[LlmCostEntity]: ...


class LiteLLMCostProvider:
    """Reads model costs from LiteLLM's bundled model_cost JSON.

    No network call is made — LiteLLM ships the cost data as a JSON file
    that is read at import time. Imported lazily to avoid startup overhead.

    LiteLLM keys many models with a redundant ``<litellm_provider>/`` prefix
    (e.g. ``azure/codex-mini`` with ``litellm_provider="azure"``, or
    ``azure/eu/gpt-4o-...``) while first-party vendors like openai/anthropic
    are keyed bare (``gpt-4o``). The prefix duplicates the ``provider`` column
    and makes model names harder for consumers to match, so we strip a single
    leading ``<provider>/`` segment — writing ``codex-mini`` /
    ``eu/gpt-4o-...`` under provider ``azure``. This mirrors what
    ``OpenRouterCostProvider`` already does with its ``<vendor>/`` prefix, and
    aligns the two sources' naming so the same model resolves to one row.

    Stripping can collapse two source keys onto one ``(provider, model)`` name
    (e.g. ``azure/computer-use-preview`` and a bare ``computer-use-preview``,
    both provider ``azure``). We de-duplicate deterministically: the collided
    pair almost always shares a price, but where it disagrees (e.g.
    ``gemini-exp-1206``: one key priced, the other ``0``) we keep the
    non-zero-priced entry — a zero price is treated as a stale/placeholder
    alias. The ``<provider>/``-prefixed (canonical) entry only breaks ties that
    price doesn't, so dict order never decides.
    """

    # LiteLLM's data file includes a `sample_spec` placeholder entry whose
    # provider is a docs URL — it documents the schema rather than a real
    # model. Drop it so it doesn't pollute the table.
    _SKIP_MODELS = frozenset({"sample_spec"})

    @staticmethod
    def _strip_provider_prefix(model_key: str, provider: str) -> str:
        """Drop a single leading ``<provider>/`` segment from a model key.

        Only the leading segment is removed, so region sub-namespaces are
        preserved (``azure/eu/gpt-4o`` -> ``eu/gpt-4o``). Keys without the
        prefix (``gpt-4o``, ``computer-use-preview``) are returned unchanged.
        """
        prefix = f"{provider}/"
        return model_key[len(prefix):] if model_key.startswith(prefix) else model_key

    def fetch_costs(self) -> list[LlmCostEntity]:
        import litellm  # noqa: PLC0415

        # Keyed by (provider, model) so prefix-stripping collisions collapse to
        # one entry. On collision the higher-ranked candidate wins (see doc):
        # rank = (has a non-zero price, key was prefixed). Stored alongside each
        # winner so the comparison is order-independent.
        by_key: dict[tuple[str, str], tuple[tuple[int, int], LlmCostEntity]] = {}

        for model_key, entry in litellm.model_cost.items():
            if model_key in self._SKIP_MODELS:
                continue

            input_cost = entry.get("input_cost_per_token")
            output_cost = entry.get("output_cost_per_token")
            provider = entry.get("litellm_provider")

            if input_cost is None or output_cost is None or not provider:
                continue

            model = self._strip_provider_prefix(model_key, provider)
            was_prefixed = model != model_key
            key = (provider, model)
            rank = (int(bool(input_cost) or bool(output_cost)), int(was_prefixed))

            existing = by_key.get(key)
            if existing is not None and rank <= existing[0]:
                continue

            by_key[key] = (
                rank,
                LlmCostEntity(
                    provider=provider,
                    model=model,
                    prompt_cost_per_token=input_cost,
                    completion_cost_per_token=output_cost,
                    price_details=_build_price_details(entry, _LITELLM_PRICE_FIELDS),
                    is_chat_capable=_litellm_is_chat_capable(entry),
                    source="litellm",
                ),
            )
        return [entity for _, entity in by_key.values()]


class OpenRouterCostProvider:
    """Fetches model costs from OpenRouter's public models API.

    OpenRouter returns ids in the form ``<vendor>/<slug>`` (e.g.
    ``anthropic/claude-opus-4.7``). We treat OpenRouter as a broad pricing
    source — not just for traffic that routes through OpenRouter — so we
    strip the vendor prefix and write rows under the native vendor's name
    (provider=anthropic, model=claude-opus-4.7).

    Variants like ``:free`` and ``:thinking`` are preserved in the model
    slug — they have distinct pricing. The ``~`` prefix on aliases like
    ``~anthropic/claude-opus-latest`` is dropped from the provider name.

    Zero-priced entries (free routes) are kept as 0.0 floats.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def fetch_costs(self) -> list[LlmCostEntity]:
        import httpx  # noqa: PLC0415

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(f"{self.base_url}/models")
            response.raise_for_status()
            payload = response.json()

        costs: list[LlmCostEntity] = []
        for entry in payload.get("data", []):
            model_id = entry.get("id")
            if not model_id or "/" not in model_id:
                continue

            vendor, slug = model_id.split("/", 1)
            # `~vendor/...` is an alias namespace (e.g. ~anthropic/claude-opus-latest).
            # Drop the tilde so the row attributes to the real vendor.
            provider = vendor.lstrip("~").lower()
            if not provider or not slug:
                continue

            pricing = entry.get("pricing") or {}
            prompt = pricing.get("prompt")
            completion = pricing.get("completion")
            if prompt is None or completion is None:
                continue

            try:
                prompt_cost = float(prompt)
                completion_cost = float(completion)
            except (TypeError, ValueError):
                continue

            costs.append(
                LlmCostEntity(
                    provider=provider,
                    model=slug,
                    prompt_cost_per_token=prompt_cost,
                    completion_cost_per_token=completion_cost,
                    price_details=_build_price_details(
                        pricing, _OPENROUTER_PRICE_FIELDS
                    ),
                    is_chat_capable=_openrouter_is_chat_capable(entry),
                    source="openrouter",
                )
            )
        return costs

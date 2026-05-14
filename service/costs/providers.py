from typing import Protocol

from service.costs.entities import LlmCostEntity


class CostDataProvider(Protocol):
    """Interface for fetching LLM cost data from an external source."""

    def fetch_costs(self) -> list[LlmCostEntity]: ...


class LiteLLMCostProvider:
    """Reads model costs from LiteLLM's bundled model_cost JSON.

    No network call is made — LiteLLM ships the cost data as a JSON file
    that is read at import time. Imported lazily to avoid startup overhead.
    """

    # LiteLLM's data file includes a `sample_spec` placeholder entry whose
    # provider is a docs URL — it documents the schema rather than a real
    # model. Drop it so it doesn't pollute the table.
    _SKIP_MODELS = frozenset({"sample_spec"})

    def fetch_costs(self) -> list[LlmCostEntity]:
        import litellm  # noqa: PLC0415

        costs = []
        for model_key, entry in litellm.model_cost.items():
            if model_key in self._SKIP_MODELS:
                continue

            input_cost = entry.get("input_cost_per_token")
            output_cost = entry.get("output_cost_per_token")
            provider = entry.get("litellm_provider")

            if input_cost is None or output_cost is None or not provider:
                continue

            costs.append(
                LlmCostEntity(
                    provider=provider,
                    model=model_key,
                    prompt_cost_per_token=input_cost,
                    completion_cost_per_token=output_cost,
                    source="litellm",
                )
            )
        return costs

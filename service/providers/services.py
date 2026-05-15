from dataclasses import dataclass

from service.costs.repositories import LlmCostRepository


class ListProviderModelsService:
    @dataclass(frozen=True)
    class Response:
        provider: str
        model_names: list[str]

    def __init__(self, repository: LlmCostRepository) -> None:
        self.repository = repository

    async def execute(self, provider: str) -> Response:
        normalized = provider.lower().strip()
        names = await self.repository.list_model_names_by_provider(provider=normalized)
        return self.Response(provider=normalized, model_names=names)

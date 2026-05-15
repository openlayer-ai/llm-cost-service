from pydantic import BaseModel


class ModelInfoSchema(BaseModel):
    name: str


class ListProviderModelsResponse(BaseModel):
    provider: str
    models: list[ModelInfoSchema]

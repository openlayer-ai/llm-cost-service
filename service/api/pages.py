from pathlib import Path

from fastapi import APIRouter, Depends, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from service.costs.repositories import LlmCostRepository
from service.costs.services import ListLlmCostsService
from service.deps import get_db_session

router = APIRouter(tags=["pages"])

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_PAGE_CACHE = "public, max-age=3600"


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_db_session),
) -> HTMLResponse:
    repository = LlmCostRepository(session)
    list_result = await ListLlmCostsService(repository).execute()
    status = await repository.get_status()

    costs = sorted(
        list_result.costs,
        key=lambda c: (c.provider.lower(), c.model.lower()),
    )
    providers = sorted({c.provider for c in costs}, key=str.lower)

    html = _templates.get_template("index.html").render(
        request=request,
        costs=costs,
        providers=providers,
        status=status,
    )
    return HTMLResponse(content=html, headers={"Cache-Control": _PAGE_CACHE})

"""Engagement dashboard routes and standalone app entrypoint."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .engagement_store import get_engagement_history

WEB_DIR = Path(__file__).resolve().parent / "web"

router = APIRouter()


@router.get("/engagement")
def engagement_page() -> FileResponse:
    return FileResponse(WEB_DIR / "engagement.html")


@router.get("/methodology")
def methodology_page() -> FileResponse:
    return FileResponse(WEB_DIR / "methodology.html")


@router.get("/api/engagement/history")
def engagement_history(limit: int = Query(default=120, ge=1, le=1000)) -> dict:
    records = get_engagement_history(limit=limit)
    latest = records[-1] if records else None
    return {"count": len(records), "latest": latest, "records": records}


def register_engagement_routes(app: FastAPI) -> None:
    app.include_router(router)


def build_app() -> FastAPI:
    app = FastAPI(title="Neurostasis Engagement")
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    register_engagement_routes(app)
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(), host="127.0.0.1", port=8001)

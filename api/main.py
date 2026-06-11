"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router as api_router
from config import load_settings
from utils.logger import get_logger, setup_logging

logger = get_logger("api.main")

_ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    settings = load_settings()
    setup_logging(settings.get("logging", {}).get("level", "INFO"))
    try:
        from knowledge_base import ingest as ingest_mod
        from retrieval.qdrant_store import QdrantStore

        store = QdrantStore.instance()
        info = store.get_collection_info(ensure=False)
        count = info.get("points_count") or 0
        if not count:
            ingest_mod.main()
        else:
            store.ensure_collection()
    except Exception as exc:  # noqa: BLE001
        logger.warning("startup_kb_init_skipped", error=str(exc))
    yield


app = FastAPI(title="CloudDash Support API", lifespan=lifespan)
app.mount("/ui", StaticFiles(directory=_ROOT / "ui", html=True), name="ui")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)


from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.error("validation_error", path=str(request.url), errors=exc.errors(), body=await request.body())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", path=str(request.url), error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "CloudDash Support", "docs": "/docs"}

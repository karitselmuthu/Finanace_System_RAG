from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.utils.config import settings
from src.utils.logger import logger

STATIC_DIR = Path(__file__).parent.parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG System starting up...")
    settings.ensure_dirs()
    yield
    logger.info("RAG System shutting down.")


app = FastAPI(
    title="RAG Transaction Analyzer",
    description="Upload PDF bank statements and query transactions using natural language.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the chat UI at "/"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
async def chat_ui():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

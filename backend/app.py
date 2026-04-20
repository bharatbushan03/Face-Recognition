from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

from backend.models.database import engine, Base
from backend.routes.face_routes import router as face_router
from backend.utils.error_handlers import setup_exception_handlers
from backend.utils.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("FR_CORS_ORIGINS", "http://localhost,http://127.0.0.1")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["http://localhost", "http://127.0.0.1"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure DB schema exists before serving requests.
    Base.metadata.create_all(bind=engine)
    logger.info("Face Recognition API starting up")
    logger.info("CORS origins: %s", _parse_cors_origins())
    yield

app = FastAPI(
    title="Face Recognition API",
    description="API for registering and recognizing faces using OpenCV and face_recognition.",
    version="1.0.0",
    lifespan=lifespan,
)

# Setup CORS (restrict by default, configurable via FR_CORS_ORIGINS env var)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup error handlers
setup_exception_handlers(app)

# Include routes
app.include_router(face_router)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    # Typically run this with: uvicorn backend.app:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)

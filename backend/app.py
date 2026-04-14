from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from backend.models.database import engine, Base
from backend.routes.face_routes import router as face_router
from backend.utils.error_handlers import setup_exception_handlers

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Face Recognition API",
    description="API for registering and recognizing faces using OpenCV and face_recognition.",
    version="1.0.0"
)

# Setup CORS to allow all for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

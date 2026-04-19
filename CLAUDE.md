# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Face Recognition web application with FastAPI backend and vanilla JS frontend. Uses SQLite for storage and the `face_recognition` library (dlib-based) for face detection and encoding.

## Commands

**Run backend:**
```bash
uvicorn backend.app:app --reload
```
API runs on `http://localhost:8000`, docs at `/docs`.

**Run tests:**
```bash
PYTHONPATH=. pytest backend/tests/           # All tests
PYTHONPATH=. pytest backend/tests/test_api.py::test_health_check  # Single test
```

**Open frontend:**
Open `frontend/index.html` in browser, or serve via `python -m http.server -d frontend/ 8080`.

## Architecture

**Backend (`backend/`)**
- `app.py` - FastAPI entry point, CORS, exception handlers, DB table initialization
- `routes/face_routes.py` - API endpoints: `POST /register`, `POST /recognize`, `GET /users`
- `services/` - Business logic: `face_service.py` (encoding extraction, comparison, smile detection), `user_service.py` (CRUD)
- `models/` - SQLAlchemy: `database.py` (connection), `user.py` (ORM with JSON-encoded face embeddings), `schemas.py` (Pydantic responses)
- `utils/` - `image_processing.py` (base64/file→RGB conversion), `error_handlers.py` (custom exceptions: `FaceNotFoundError`, `MultipleFacesError`)

**Frontend (`frontend/`)**
- `index.html` - UI with video feed, HUD canvas overlay, toast notifications
- `script.js` - Camera capture, API calls, voice synthesis, HUD rendering

**Data flow:**
1. Frontend captures video frame → base64 → FormData POST
2. Backend converts to RGB numpy array via OpenCV → extracts 128-d face encoding
3. SQLite stores encodings as JSON arrays; comparison uses `face_recognition.face_distance()` with tolerance 0.5
4. Smile detection via facial landmarks (mouth width / face width ratio > 0.38)

**Database:**
- Location: `backend/database/faces.db` (auto-created)
- Single table: `users` (id, name, face_encoding JSON, created_at)

## Key Dependencies

See `requirements.txt`: FastAPI, SQLAlchemy, face_recognition, opencv-python-headless, numpy, pytest, httpx, python-multipart.

## Testing Notes

- Tests use in-memory SQLite with `StaticPool`
- `test_api.py` tests health check, face comparison algorithm, and error handling
- Full endpoint tests require base64 face images or mocked services

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
PYTHONPATH=. pytest backend/tests/
```

**Open frontend:**
Open `frontend/index.html` in browser, or serve via `python -m http.server -d frontend/ 8080`.

## Architecture

**Backend (`backend/`)**
- `app.py` - FastAPI entry point, CORS, exception handlers
- `routes/face_routes.py` - API endpoints: `/register`, `/recognize`, `/users`
- `services/` - Business logic: `face_service.py` (encoding extraction, comparison), `user_service.py` (CRUD)
- `models/` - SQLAlchemy: `database.py` (connection), `user.py` (ORM), `schemas.py` (Pydantic)
- `utils/` - `image_processing.py` (base64/file→RGB), `error_handlers.py` (custom exceptions)

**Frontend (`frontend/`)**
- `index.html` - UI with video feed, HUD canvas overlay
- `script.js` - Camera capture, API calls, voice synthesis, HUD rendering

**Data flow:**
1. Frontend captures video frame → base64
2. Backend converts to RGB numpy array → extracts 128-d face encoding
3. SQLite stores encodings as JSON; comparison uses `face_recognition.face_distance()`
4. Smile detection via facial landmarks (mouth width / face width ratio > 0.38)

## Key Dependencies

See `requirements.txt`: FastAPI, SQLAlchemy, face_recognition, opencv-python-headless, pytest, httpx.

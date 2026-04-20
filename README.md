# Face Recognition Attendance System

Production-ready face recognition attendance system with FastAPI backend, real-time webcam pipeline, admin tools, and Streamlit dashboard.

## 1. What This Project Does

- Face detection and real-time tracking
- Face recognition with configurable thresholding and ambiguity handling
- Attendance logging with duplicate protection (day/session scope)
- Admin module for student dataset management and encoding rebuilds
- Streamlit dashboard and browser frontend for operations/demo

## 2. Architecture

### Backend modules
- `backend/services/realtime_face_detection.py`: detection + smoothing + tracking IDs
- `backend/services/realtime_face_recognition.py`: batch encoding + batch matching + low-light and quality filters
- `backend/services/face_encoding_store.py`: encoding cache, vectorized matching, dataset build
- `backend/services/attendance_service.py`: durable CSV attendance with dedupe policy
- `backend/services/admin_service.py`: add/update/remove students and rebuild cache

### Interfaces
- API: `backend/app.py`, routes in `backend/routes/face_routes.py`
- Realtime CLI scripts in `backend/scripts/`
- Dashboard: `backend/ui/streamlit_dashboard.py`
- Frontend: `frontend/`

## 3. Setup

### Prerequisites
- Python 3.10+
- Webcam (for realtime mode)

### Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Environment

Copy `.env.example` to `.env` and adjust values:

```bash
# Windows PowerShell
Copy-Item .env.example .env
```

Key options:
- `FR_LOG_LEVEL`: logging verbosity (`INFO`, `DEBUG`, ...)
- `FR_CORS_ORIGINS`: comma-separated allowed frontend origins
- `FR_RECOGNITION_TOLERANCE`: API match threshold
- `FR_ADMIN_USERNAME`, `FR_ADMIN_PASSWORD`: optional basic auth for admin API routes

## 4. Running the System

### Minimal startup (recommended)

Start these 2 processes:

```bash
uvicorn backend.app:app --reload
streamlit run backend/ui/streamlit_dashboard.py
```

This is enough for normal operation. The dashboard now boots even when no students are registered yet.
Register users through the API/UI flow, then click **Rebuild Face Cache** in Streamlit.

### API

```bash
uvicorn backend.app:app --reload
```

- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### Browser frontend

```bash
python -m http.server -d frontend 8080
```

Then open `http://localhost:8080`.

### Streamlit dashboard

```bash
streamlit run backend/ui/streamlit_dashboard.py
```

Use the in-app tuning controls for threshold, frame skip, max faces/frame, and low-light enhancement.

### Realtime recognition CLI

```bash
python -m backend.scripts.run_realtime_face_recognition \
  --dataset-dir data/students \
  --cache-file models/known_face_encodings.npz \
  --detect-every-n-frames 2 \
  --process-every-n-frames 2 \
  --detect-process-width 640 \
  --max-faces-per-frame 6
```

### Realtime benchmark CLI

```bash
python -m backend.scripts.benchmark_realtime_pipeline \
  --dataset-dir data/students \
  --cache-file models/known_face_encodings.npz \
  --frames 300 \
  --detect-every-n-frames 2 \
  --process-every-n-frames 2 \
  --detect-process-width 640
```

This command reports effective FPS plus mean/P50/P95 frame processing latency.

### Admin CLI

```bash
python -m backend.scripts.admin_cli
```

## 5. Performance Optimization (Applied)

The pipeline is optimized for practical classroom usage:

- Batch face encodings per frame instead of one call per face
- Vectorized batch distance matching in `KnownFaceStore.match_batch`
- Detection skipping (`detect_every_n_frames`) to reuse recent boxes between frames
- Frame skipping (`process_every_n_frames`) for controlled CPU usage
- Detection resizing (`process_width`) for faster detector inference
- Camera buffer tuning (`CAP_PROP_BUFFERSIZE`) to reduce lag
- Optional multithreaded fallback for per-face encoding edge cases
- Face quality gating (brightness/sharpness) to avoid expensive low-value matches

Recommended baseline:
- `--detect-every-n-frames 2`
- `--process-every-n-frames 2`
- `--detect-process-width 640`
- `--max-faces-per-frame 4..8` based on hardware

## 6. Accuracy Improvements (Applied + Practical Guidance)

Implemented:
- Adjustable tolerance/ambiguity margin
- Low-light enhancement using CLAHE before recognition
- Low-quality-face rejection (very dark/blurred crops)

Recommended tuning process:
1. Start tolerance at `0.48`.
2. Increase to `0.52` if too many false negatives.
3. Decrease to `0.45` if false positives appear.
4. Keep multiple images per student with varied angles/lighting.

Occlusion and angle handling:
- Add masked/glasses samples per student in dataset.
- Prefer high-quality enrollment images.
- Use `encode_model=large` for higher accuracy if hardware allows.

## 7. Logging, Monitoring, Stability

Implemented:
- Centralized logging config (`backend/utils/logging_config.py`)
- Rotating log file output (`logs/system.log`)
- Attendance event logs via dedicated logger
- Camera read failure retry windows in realtime and dashboard runtime
- Corrupted attendance CSV backup and auto-healing
- Corrupted encoding cache fallback/rebuild paths

## 8. Security Considerations

Implemented:
- Optional HTTP Basic Auth for admin routes (`/api/face/register`, `/api/face/users`)
- Safer CORS defaults (local origins by default)
- `.env.example` uses placeholders only (no secrets)

Recommendations:
- Set strong `FR_ADMIN_PASSWORD` for demos/public networks.
- Restrict `FR_CORS_ORIGINS` to known frontend domains.
- Keep attendance files and model cache on protected storage.

## 9. Deployment Options

### Local classroom PC (recommended)
- Run FastAPI + Streamlit locally.
- Use webcam from the host machine.
- Store attendance CSV locally.

### Web deployment (API + dashboard)
- FastAPI on VM/container (Render, Fly.io, Azure, AWS, GCP).
- Streamlit Cloud for dashboard-only monitoring.
- For cloud webcam capture, use client-side capture and API upload.

### Edge device (Raspberry Pi / mini-PC)
- Lower camera resolution and increase frame skipping.
- Keep `process_width` conservative (`320..640`).
- Consider offloading recognition to server if Pi CPU is constrained.

## 10. Advanced Features (Roadmap)

Useful next additions:
- Email/SMS attendance notifications
- Mask detection classifier for policy compliance
- Attendance analytics charts and trend reports
- Multi-camera ingestion with camera IDs in attendance records
- Cloud database integration (PostgreSQL/MongoDB) for centralized records

## 11. Testing & Validation

### Run tests

```bash
PYTHONPATH=. pytest backend/tests/
```

### End-to-end validation checklist
1. Register at least 10 students with varied lighting/angles.
2. Run recognition session for 10-15 minutes.
3. Validate no duplicate attendance rows in chosen dedupe scope.
4. Verify behavior for unknown faces, blur, and low-light scenes.
5. Restart app and confirm attendance state and cache loading are stable.

### Performance benchmarking suggestions
1. Run the benchmark CLI at least 3 times and average FPS/latency metrics.
2. Compare `detect_every_n_frames` values: `1`, `2`, `3`.
3. Compare `process_every_n_frames` values: `1`, `2`, `3`.
4. Compare `detect_process_width` values: `320`, `640`, `960`.
5. Record CPU and memory usage under expected class size.

## 12. Final Demo/Submission Checklist

1. `.env` configured and secrets not hardcoded.
2. Dataset has multiple quality images per student.
3. Encoding cache rebuilt after final dataset update.
4. Camera access tested on target machine.
5. API health endpoint returns OK.
6. Attendance CSV write permissions verified.
7. Logs rotate and are writable (`logs/system.log`).
8. Admin endpoints protected if running beyond localhost.
9. End-to-end smoke test completed.
10. Backup plan ready (fallback to API-only or dashboard-only mode).

## Common Mistakes to Avoid

- Using too high tolerance and accepting false matches
- Enrolling students with only one image
- Running on incorrect Python environment/interpreter
- Keeping permissive CORS (`*`) in shared/public deployments
- Ignoring camera latency from high frame size + large buffer

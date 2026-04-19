# Face Recognition Application

A complete, modern Face Recognition web application.

## Features
- Clean Architecture (FastAPI backend + Vanilla JS frontend)
- Local SQLite Database to store users and encodings
- Robust error handling for multiple faces, no faces, missing data, etc.
- Responsive Dark Mode UI with micro-animations
- High confidence accurate face matching

## Tech Stack
- **Backend:** Python, FastAPI, SQLAlchemy, `face_recognition` (dlib), OpenCV
- **Frontend:** Vanilla HTML/CSS/JS
- **Database:** SQLite

## How to Run

1. **Activate Environment**
   ```bash
   source venv/bin/activate
   ```

2. **Start Backend Server**
   Run the backend from the root repo directory:
   ```bash
   uvicorn backend.app:app --reload
   ```
   The backend API will run on `http://localhost:8000`. Auto-generated API docs are at `http://localhost:8000/docs`.

3. **Start Frontend**
   Simply open `frontend/index.html` in your web browser. Alternatively, run a simple HTTP server:
   ```bash
   python -m http.server -d frontend/ 8080
   ```
   And visit `http://localhost:8080`.

## Testing

Run unit tests via:
```bash
source venv/bin/activate
PYTHONPATH=. pytest backend/tests/
```

## Streamlit Dashboard UI

Run the live attendance dashboard:

```bash
streamlit run backend/ui/streamlit_dashboard.py
```

Dashboard capabilities:
- Live webcam feed with face detection and recognition overlays
- Real-time attendance table with search
- Start/Stop camera controls
- Total students marked present today
- CSV export button for attendance logs

Quick verification checklist:
1. Start camera and confirm live feed appears in the dashboard.
2. Verify recognized names are overlaid on the video frame.
3. Confirm new attendance entries appear in the table immediately.
4. Check the "Total Present Today" metric increases when new students are marked.

## Common Issues
- **Multiple Faces Error**: Ensure only one person is in the frame when registering or acknowledging a face.
- **Camera Error**: Your browser must give camera permissions to `index.html`. If accessing over the network (not localhost), you might need HTTPS. Localhost is permitted by default.

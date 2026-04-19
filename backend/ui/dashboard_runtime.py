from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import csv
import platform

import cv2
import numpy as np

from backend.services.attendance_service import AttendanceConfig, AttendanceManager
from backend.services.face_encoding_store import KnownFaceStore, build_known_face_store
from backend.services.realtime_face_detection import FaceDetectionConfig, MediaPipeFaceDetector
from backend.services.realtime_face_recognition import (
    RecognitionConfig,
    RealtimeFaceRecognizer,
    draw_recognition_results,
)


@dataclass(frozen=True)
class DashboardRuntimeConfig:
    dataset_dir: str = "data/students"
    cache_file: str = "models/known_face_encodings.npz"
    refresh_cache: bool = False
    attendance_file: str = "attendance_logs/attendance.csv"
    dedupe_scope: str = "day"
    session_id: str | None = None
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    mirror: bool = True
    display_unknown_warnings: bool = False


@dataclass(frozen=True)
class FrameUpdate:
    frame_rgb: np.ndarray
    total_faces: int
    known_faces: int
    unknown_faces: int
    new_attendance_events: list[dict[str, str]]
    notice: str | None


class AttendanceDashboardRuntime:
    """Orchestrates camera capture, face recognition, and attendance persistence."""

    def __init__(self, config: DashboardRuntimeConfig | None = None):
        self.config = config or DashboardRuntimeConfig()

        self.dataset_dir = Path(self.config.dataset_dir)
        self.cache_file = Path(self.config.cache_file)
        self.attendance_file = Path(self.config.attendance_file)

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.attendance_file.parent.mkdir(parents=True, exist_ok=True)

        self._known_faces = self._load_or_build_known_faces()
        self.detector = MediaPipeFaceDetector(
            FaceDetectionConfig(
                process_width=640,
                min_detection_confidence=0.65,
                model_selection=0,
                smoothing_alpha=0.65,
                max_tracking_distance=90.0,
                max_missing_frames=5,
                min_face_size=28,
            )
        )
        self.recognizer = RealtimeFaceRecognizer(
            known_faces=self._known_faces,
            detector=self.detector,
            config=RecognitionConfig(
                tolerance=0.48,
                ambiguity_margin=0.03,
                process_every_n_frames=2,
                encode_model="small",
            ),
        )

        self.attendance = AttendanceManager(
            AttendanceConfig(
                csv_path=str(self.attendance_file),
                dedupe_scope=self.config.dedupe_scope,
                session_id=self.config.session_id,
            )
        )

        self._cap: cv2.VideoCapture | None = None
        self._present_today_keys: set[str] = self._load_present_today_keys()

    def _load_or_build_known_faces(self) -> KnownFaceStore:
        if self.cache_file.exists() and not self.config.refresh_cache:
            known_faces = KnownFaceStore.load_npz(self.cache_file)
            if known_faces.size > 0:
                return known_faces

        known_faces, _report = build_known_face_store(dataset_dir=self.dataset_dir, detection_model="hog", num_jitters=1)
        known_faces.save_npz(self.cache_file)

        if known_faces.size == 0:
            raise RuntimeError(
                "No known face encodings found. Add student images in data/students and rebuild cache."
            )

        return known_faces

    def _open_camera(self) -> cv2.VideoCapture:
        if platform.system().lower() == "windows":
            cap = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap

        return cv2.VideoCapture(self.config.camera_index)

    def start_camera(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return

        cap = self._open_camera()
        if not cap.isOpened():
            raise RuntimeError(
                "Could not access webcam. Check camera permissions and close other apps using the camera."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self._cap = cap

    def stop_camera(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_camera_running(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def _load_present_today_keys(self) -> set[str]:
        if not self.attendance_file.exists() or self.attendance_file.stat().st_size == 0:
            return set()

        today = datetime.now().date().isoformat()
        keys: set[str] = set()

        with self.attendance_file.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("date") != today:
                    continue

                key = (row.get("student_key") or "").strip()
                if key:
                    keys.add(key)

        return keys

    @property
    def total_present_today(self) -> int:
        return len(self._present_today_keys)

    @property
    def detection_backend(self) -> str:
        return self.detector.backend_name

    def process_next_frame(self) -> FrameUpdate:
        if not self.is_camera_running():
            raise RuntimeError("Camera is not running. Start the camera first.")

        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Unable to read webcam frame.")

        if self.config.mirror:
            frame = cv2.flip(frame, 1)

        results = self.recognizer.recognize_frame(frame)

        matched_in_frame: dict[str, float] = {}
        for item in results:
            if item.status != "matched":
                continue

            previous_conf = matched_in_frame.get(item.name)
            if previous_conf is None or item.confidence > previous_conf:
                matched_in_frame[item.name] = item.confidence

        new_events: list[dict[str, str]] = []
        for student_name, confidence in matched_in_frame.items():
            marked, record = self.attendance.mark_present(
                student_name=student_name,
                confidence=confidence,
                source="streamlit_dashboard",
            )
            if marked and record is not None:
                self._present_today_keys.add(record.student_key)
                new_events.append(
                    {
                        "student_name": record.student_name,
                        "date": record.date,
                        "time": record.time,
                        "recorded_at": record.recorded_at,
                    }
                )

        draw_recognition_results(frame, results)

        total_faces = len(results)
        known_faces = sum(1 for result in results if result.status == "matched")
        unknown_faces = total_faces - known_faces

        notice: str | None = None
        if total_faces == 0:
            notice = "No faces detected in current frame."
        elif unknown_faces > 0 and self.config.display_unknown_warnings:
            notice = "Some faces were detected but not matched to known students."

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return FrameUpdate(
            frame_rgb=frame_rgb,
            total_faces=total_faces,
            known_faces=known_faces,
            unknown_faces=unknown_faces,
            new_attendance_events=new_events,
            notice=notice,
        )

    def get_attendance_rows(
        self,
        search_query: str = "",
        highlight_seconds: int = 20,
        max_rows: int = 500,
    ) -> list[dict[str, str]]:
        if not self.attendance_file.exists() or self.attendance_file.stat().st_size == 0:
            return []

        query = search_query.strip().lower()
        now = datetime.now()

        rows: list[dict[str, str]] = []
        with self.attendance_file.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                student_name = (row.get("student_name") or "").strip()
                student_id = (row.get("student_id") or "").strip()

                haystack = f"{student_name} {student_id}".lower()
                if query and query not in haystack:
                    continue

                recorded_at = (row.get("recorded_at") or "").strip()
                is_new = ""
                if recorded_at:
                    try:
                        event_time = datetime.fromisoformat(recorded_at)
                        if now - event_time <= timedelta(seconds=highlight_seconds):
                            is_new = "NEW"
                    except ValueError:
                        pass

                rows.append(
                    {
                        "New": is_new,
                        "Student Name": student_name,
                        "Student ID": student_id,
                        "Date": (row.get("date") or "").strip(),
                        "Time": (row.get("time") or "").strip(),
                    }
                )

        rows.reverse()
        if max_rows > 0:
            return rows[:max_rows]
        return rows

    def get_attendance_csv_bytes(self) -> bytes | None:
        if not self.attendance_file.exists() or self.attendance_file.stat().st_size == 0:
            return None
        return self.attendance_file.read_bytes()

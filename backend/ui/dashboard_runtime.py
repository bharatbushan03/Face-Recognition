from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import csv
import inspect
import logging
import platform

import cv2
import numpy as np

from backend.models.database import SessionLocal
from backend.models.user import User
from backend.services.attendance_service import AttendanceConfig, AttendanceManager
from backend.services.face_encoding_store import KnownFaceStore, build_known_face_store
from backend.services.realtime_face_detection import FaceDetectionConfig, MediaPipeFaceDetector
from backend.services.realtime_face_recognition import (
    RecognizedFace,
    RecognitionConfig,
    RealtimeFaceRecognizer,
    draw_recognition_results,
)
from backend.utils.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DashboardRuntimeConfig:
    dataset_dir: str = "data/students"
    cache_file: str = "models/known_face_encodings.npz"
    refresh_cache: bool = False
    attendance_file: str = "attendance_logs/attendance.csv"
    dedupe_scope: str = "day"
    session_id: str | None = None
    camera_index: int = 0
    camera_buffer_size: int = 1
    frame_width: int = 960
    frame_height: int = 540
    mirror: bool = True
    display_unknown_warnings: bool = False
    display_smoothing_alpha: float = 0.0
    max_read_failures: int = 15
    detection_process_width: int = 640
    detection_min_confidence: float = 0.65
    recognition_tolerance: float = 0.48
    recognition_ambiguity_margin: float = 0.03
    recognition_detect_every_n_frames: int = 2
    recognition_process_every_n_frames: int = 2
    recognition_encode_model: str = "small"
    recognition_max_faces_per_frame: int = 6
    recognition_enable_low_light_enhancement: bool = True
    recognition_low_light_threshold: float = 70.0
    recognition_min_face_brightness: float = 35.0
    recognition_min_face_sharpness: float = 12.0


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
        if self.config.recognition_detect_every_n_frames <= 0:
            raise ValueError("recognition_detect_every_n_frames must be > 0")

        self.dataset_dir = Path(self.config.dataset_dir)
        self.cache_file = Path(self.config.cache_file)
        self.attendance_file = Path(self.config.attendance_file)

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.attendance_file.parent.mkdir(parents=True, exist_ok=True)

        self.detector = MediaPipeFaceDetector(
            FaceDetectionConfig(
                process_width=self.config.detection_process_width,
                min_detection_confidence=self.config.detection_min_confidence,
                model_selection=0,
                smoothing_alpha=0.65,
                max_tracking_distance=90.0,
                max_missing_frames=5,
                min_face_size=28,
            )
        )
        self._recognition_config = self._build_recognition_config()

        self._known_faces = KnownFaceStore.empty()
        self.recognizer = RealtimeFaceRecognizer(
            known_faces=self._known_faces,
            detector=self.detector,
            config=self._recognition_config,
        )
        self.sync_known_faces(force_refresh_cache=self.config.refresh_cache)

        self.attendance = AttendanceManager(
            AttendanceConfig(
                csv_path=str(self.attendance_file),
                dedupe_scope=self.config.dedupe_scope,
                session_id=self.config.session_id,
            )
        )

        self._cap: cv2.VideoCapture | None = None
        self._present_today_keys: set[str] = self._load_present_today_keys()
        self._consecutive_read_failures = 0
        self._last_frame_rgb: np.ndarray | None = None
        self._last_display_bgr: np.ndarray | None = None
        self._dashboard_frame_index = 0
        self._last_unknown_results: list[RecognizedFace] = []

    def _build_recognition_config(self) -> RecognitionConfig:
        raw_kwargs = {
            "tolerance": self.config.recognition_tolerance,
            "ambiguity_margin": self.config.recognition_ambiguity_margin,
            "detect_every_n_frames": self.config.recognition_detect_every_n_frames,
            "process_every_n_frames": self.config.recognition_process_every_n_frames,
            "encode_model": self.config.recognition_encode_model,
            "max_faces_per_frame": self.config.recognition_max_faces_per_frame,
            "enable_low_light_enhancement": self.config.recognition_enable_low_light_enhancement,
            "low_light_threshold": self.config.recognition_low_light_threshold,
            "min_face_brightness": self.config.recognition_min_face_brightness,
            "min_face_sharpness": self.config.recognition_min_face_sharpness,
        }

        accepted_params = set(inspect.signature(RecognitionConfig).parameters)
        filtered_kwargs = {
            key: value
            for key, value in raw_kwargs.items()
            if key in accepted_params
        }

        if len(filtered_kwargs) != len(raw_kwargs):
            missing_keys = sorted(set(raw_kwargs) - set(filtered_kwargs))
            logger.warning(
                "RecognitionConfig does not accept %s. Falling back to compatible args only.",
                ", ".join(missing_keys),
            )

        return RecognitionConfig(**filtered_kwargs)

    def _load_known_faces_from_database(self) -> KnownFaceStore:
        names: list[str] = []
        encodings: list[np.ndarray] = []

        db = SessionLocal()
        try:
            users = db.query(User).all()
            for user in users:
                try:
                    encoding = np.asarray(user.get_encoding(), dtype=np.float32).reshape(-1)
                except Exception as exc:
                    logger.warning("Skipping user %s due to invalid encoding (%s)", user.name, exc)
                    continue

                if encoding.shape != (128,):
                    logger.warning(
                        "Skipping user %s due to unexpected encoding shape: %s",
                        user.name,
                        encoding.shape,
                    )
                    continue

                names.append(user.name)
                encodings.append(encoding)
        except Exception as exc:
            logger.warning("Failed reading known faces from database: %s", exc)
        finally:
            db.close()

        if not names:
            return KnownFaceStore.empty()

        return KnownFaceStore(names=names, encodings=np.vstack(encodings))

    def _load_or_build_known_faces(self, refresh_cache: bool | None = None) -> KnownFaceStore:
        should_refresh = self.config.refresh_cache if refresh_cache is None else refresh_cache

        if self.cache_file.exists() and not should_refresh:
            try:
                known_faces = KnownFaceStore.load_npz(self.cache_file)
                if known_faces.size > 0:
                    return known_faces
            except Exception as exc:
                logger.warning(
                    "Unable to load face cache from %s (%s). Trying database and dataset sources.",
                    self.cache_file,
                    exc,
                )

        db_known_faces = self._load_known_faces_from_database()
        if db_known_faces.size > 0:
            db_known_faces.save_npz(self.cache_file)
            return db_known_faces

        if self.dataset_dir.exists():
            try:
                known_faces, _report = build_known_face_store(
                    dataset_dir=self.dataset_dir,
                    detection_model="hog",
                    num_jitters=1,
                )
                if known_faces.size > 0:
                    known_faces.save_npz(self.cache_file)
                    return known_faces
            except Exception as exc:
                logger.warning("Failed building known face store from dataset: %s", exc)

        logger.info(
            "No known face encodings available yet. Register students via FastAPI and then refresh cache."
        )
        return KnownFaceStore.empty()

    def sync_known_faces(self, force_refresh_cache: bool = False) -> int:
        self._known_faces = self._load_or_build_known_faces(refresh_cache=force_refresh_cache)
        self.recognizer = RealtimeFaceRecognizer(
            known_faces=self._known_faces,
            detector=self.detector,
            config=self._recognition_config,
        )
        return self._known_faces.size

    @property
    def known_faces_count(self) -> int:
        return self._known_faces.size

    def _open_camera(self) -> cv2.VideoCapture:
        if platform.system().lower() == "windows":
            cap = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                if self.config.camera_buffer_size > 0:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.camera_buffer_size)
                return cap

        cap = cv2.VideoCapture(self.config.camera_index)
        if cap.isOpened() and self.config.camera_buffer_size > 0:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.camera_buffer_size)
        return cap

    def start_camera(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return

        # Pull latest known users before opening camera so new registrations are recognized.
        self.sync_known_faces(force_refresh_cache=False)

        cap = self._open_camera()
        if not cap.isOpened():
            raise RuntimeError(
                "Could not access webcam. Check camera permissions and close other apps using the camera."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self._cap = cap
        self._consecutive_read_failures = 0
        self._last_frame_rgb = None
        self._last_display_bgr = None
        self._dashboard_frame_index = 0
        self._last_unknown_results = []

    def stop_camera(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._consecutive_read_failures = 0
        self._last_frame_rgb = None
        self._last_display_bgr = None
        self._dashboard_frame_index = 0
        self._last_unknown_results = []

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

    def _blend_display_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Blend a small amount of the previous frame into the current display frame."""
        smoothing_alpha = min(max(float(self.config.display_smoothing_alpha), 0.0), 0.95)
        display_frame = frame_bgr.copy()

        if (
            smoothing_alpha > 0.0
            and self._last_display_bgr is not None
            and self._last_display_bgr.shape == display_frame.shape
        ):
            # The smoothing control represents how much of the previous frame we keep.
            display_frame = cv2.addWeighted(
                display_frame,
                1.0 - smoothing_alpha,
                self._last_display_bgr,
                smoothing_alpha,
                0.0,
            )

        self._last_display_bgr = display_frame.copy()
        return display_frame

    def _detect_unknown_faces(self, frame_bgr: np.ndarray) -> list[RecognizedFace]:
        should_detect = (
            not self._last_unknown_results
            or (self._dashboard_frame_index % self.config.recognition_detect_every_n_frames) == 0
        )
        if not should_detect:
            return [
                RecognizedFace(
                    track_id=item.track_id,
                    bbox=item.bbox,
                    name=item.name,
                    confidence=item.confidence,
                    distance=item.distance,
                    status=item.status,
                )
                for item in self._last_unknown_results
            ]

        detections = self.detector.detect(frame_bgr)
        if len(detections) > self.config.recognition_max_faces_per_frame:
            detections = sorted(detections, key=lambda item: item.score, reverse=True)[
                : self.config.recognition_max_faces_per_frame
            ]
        results = [
            RecognizedFace(
                track_id=face.track_id,
                bbox=face.bbox,
                name="Unknown",
                confidence=0.0,
                distance=1.0,
                status="unknown",
            )
            for face in detections
        ]
        self._last_unknown_results = list(results)
        return results

    def process_next_frame(self) -> FrameUpdate:
        if not self.is_camera_running():
            raise RuntimeError("Camera is not running. Start the camera first.")

        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok:
            self._consecutive_read_failures += 1
            logger.warning(
                "Dashboard camera read failed (%s/%s)",
                self._consecutive_read_failures,
                self.config.max_read_failures,
            )
            if self._consecutive_read_failures >= self.config.max_read_failures:
                raise RuntimeError("Unable to read webcam frames consistently. Check camera connection.")

            fallback = self._last_frame_rgb
            if fallback is None:
                fallback = np.zeros((self.config.frame_height, self.config.frame_width, 3), dtype=np.uint8)

            return FrameUpdate(
                frame_rgb=fallback.copy(),
                total_faces=0,
                known_faces=0,
                unknown_faces=0,
                new_attendance_events=[],
                notice="Temporary camera read failure. Retrying...",
            )

        self._consecutive_read_failures = 0

        if self.config.mirror:
            frame = cv2.flip(frame, 1)

        self._dashboard_frame_index += 1

        no_registered_users = self._known_faces.size == 0
        if no_registered_users:
            results = self._detect_unknown_faces(frame)
        else:
            self._last_unknown_results = []
            results = self.recognizer.recognize_frame(frame)

        display_frame = self._blend_display_frame(frame)

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
                logger.info(
                    "Dashboard attendance marked for %s at %s",
                    record.student_name,
                    record.time,
                )
                new_events.append(
                    {
                        "student_name": record.student_name,
                        "date": record.date,
                        "time": record.time,
                        "recorded_at": record.recorded_at,
                    }
                )

        draw_recognition_results(display_frame, results)

        total_faces = len(results)
        known_faces = sum(1 for result in results if result.status == "matched")
        unknown_faces = total_faces - known_faces

        notice: str | None = None
        if no_registered_users:
            notice = "No registered users found. Use FastAPI /api/face/register, then refresh cache."
        elif total_faces == 0:
            notice = "No faces detected in current frame."
        elif unknown_faces > 0 and self.config.display_unknown_warnings:
            notice = "Some faces were detected but not matched to known students."

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        self._last_frame_rgb = frame_rgb.copy()
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

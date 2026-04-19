from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
import sys

import cv2
import numpy as np


Box = Tuple[int, int, int, int]  # x, y, width, height


@dataclass(frozen=True)
class DetectedFace:
    """A single face detection result for one frame."""

    track_id: int
    bbox: Box
    score: float


@dataclass
class FaceDetectionConfig:
    """Configuration for real-time face detection."""

    process_width: int = 640
    min_detection_confidence: float = 0.65
    model_selection: int = 0
    smoothing_alpha: float = 0.65
    max_tracking_distance: float = 90.0
    max_missing_frames: int = 5
    min_face_size: int = 28


@dataclass
class _TrackState:
    bbox: Tuple[float, float, float, float]
    last_seen: int


class _TemporalBoxSmoother:
    """Smooths bounding boxes across frames by matching nearby detections."""

    def __init__(self, alpha: float, max_distance: float, max_missing_frames: int):
        self.alpha = float(alpha)
        self.max_distance = float(max_distance)
        self.max_missing_frames = int(max_missing_frames)
        self.frame_index = 0
        self.next_track_id = 1
        self.tracks: dict[int, _TrackState] = {}

    @staticmethod
    def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x, y, w, h = bbox
        return x + (w / 2.0), y + (h / 2.0)

    @staticmethod
    def _to_int_box(bbox: Tuple[float, float, float, float]) -> Box:
        x, y, w, h = bbox
        return int(round(x)), int(round(y)), int(round(w)), int(round(h))

    def update(self, detections: List[Tuple[Box, float]]) -> List[DetectedFace]:
        """
        Update tracked boxes and return smoothed detections for the current frame.

        Args:
            detections: List[(bbox, score)] where bbox is (x, y, w, h)
        """
        self.frame_index += 1
        used_detection_indices: set[int] = set()
        assignments: dict[int, int] = {}  # detection_idx -> track_id

        # Greedy center-based matching between current detections and existing tracks.
        for track_id, track in list(self.tracks.items()):
            track_cx, track_cy = self._center(track.bbox)
            best_det_idx = -1
            best_distance = self.max_distance

            for det_idx, (bbox, _score) in enumerate(detections):
                if det_idx in used_detection_indices:
                    continue

                det_cx, det_cy = self._center(tuple(float(v) for v in bbox))
                distance = math.hypot(det_cx - track_cx, det_cy - track_cy)

                if distance < best_distance:
                    best_distance = distance
                    best_det_idx = det_idx

            if best_det_idx >= 0:
                used_detection_indices.add(best_det_idx)
                assignments[best_det_idx] = track_id

        results: List[DetectedFace] = []

        for det_idx, (bbox, score) in enumerate(detections):
            box_f = tuple(float(v) for v in bbox)

            if det_idx in assignments:
                track_id = assignments[det_idx]
                prev = self.tracks[track_id].bbox
                smoothed = tuple(
                    (self.alpha * new_v) + ((1.0 - self.alpha) * old_v)
                    for new_v, old_v in zip(box_f, prev)
                )
                self.tracks[track_id] = _TrackState(bbox=smoothed, last_seen=self.frame_index)
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = _TrackState(bbox=box_f, last_seen=self.frame_index)

            results.append(
                DetectedFace(
                    track_id=track_id,
                    bbox=self._to_int_box(self.tracks[track_id].bbox),
                    score=float(score),
                )
            )

        # Remove tracks not seen for a few frames to keep state small and stable.
        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if (self.frame_index - track.last_seen) > self.max_missing_frames
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

        return results


class MediaPipeFaceDetector:
    """
    Real-time face detector with temporal stabilization.

    Preferred backend: MediaPipe solutions face detector.
    Fallback backend: OpenCV Haar cascade (for environments where mediapipe
    exposes only tasks API, such as some Python 3.13 builds).
    """

    def __init__(self, config: FaceDetectionConfig | None = None):
        self.config = config or FaceDetectionConfig()

        if self.config.process_width <= 0:
            raise ValueError("process_width must be > 0")
        if not (0.0 < self.config.min_detection_confidence <= 1.0):
            raise ValueError("min_detection_confidence must be in (0.0, 1.0]")

        self._mp = None
        self._detector = None
        self._cascade = None
        self._backend_name = "opencv_haar"

        # In common Python 3.13 setups, MediaPipe exposes only mp.tasks and
        # not mp.solutions. Skip eager import there and use the fallback backend.
        if sys.version_info < (3, 13):
            try:
                import mediapipe as mp
                self._mp = mp
            except Exception:
                self._mp = None

        # Preferred path for environments that still expose mp.solutions.
        if (
            self._mp is not None
            and hasattr(self._mp, "solutions")
            and hasattr(self._mp.solutions, "face_detection")
        ):
            self._detector = self._mp.solutions.face_detection.FaceDetection(
                model_selection=self.config.model_selection,
                min_detection_confidence=self.config.min_detection_confidence,
            )
            self._backend_name = "mediapipe_solutions"
        else:
            # Safe fallback for newer MediaPipe wheels that do not expose mp.solutions.
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
            if self._cascade.empty():
                raise RuntimeError(
                    "OpenCV Haar cascade backend is unavailable. "
                    "Ensure opencv-python is installed correctly."
                )

        self._smoother = _TemporalBoxSmoother(
            alpha=self.config.smoothing_alpha,
            max_distance=self.config.max_tracking_distance,
            max_missing_frames=self.config.max_missing_frames,
        )

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @staticmethod
    def _clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Box | None:
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2 - x1, y2 - y1

    def _prepare_processing_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize the frame for faster inference.

        Returns:
            processing_frame, scale_ratio
            scale_ratio = processing_width / original_width
        """
        original_h, original_w = frame_bgr.shape[:2]

        if original_w <= self.config.process_width:
            return frame_bgr, 1.0

        scale = self.config.process_width / float(original_w)
        new_h = max(1, int(original_h * scale))
        resized = cv2.resize(frame_bgr, (self.config.process_width, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale

    def _detect_raw(self, frame_bgr: np.ndarray) -> List[Tuple[Box, float]]:
        frame_h, frame_w = frame_bgr.shape[:2]
        detections: List[Tuple[Box, float]] = []

        if self._backend_name == "mediapipe_solutions":
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._detector.process(frame_rgb)

            if not result.detections:
                return detections

            for detection in result.detections:
                rel = detection.location_data.relative_bounding_box

                x1 = int(rel.xmin * frame_w)
                y1 = int(rel.ymin * frame_h)
                x2 = int((rel.xmin + rel.width) * frame_w)
                y2 = int((rel.ymin + rel.height) * frame_h)

                clipped = self._clip_box(x1, y1, x2, y2, frame_w, frame_h)
                if clipped is None:
                    continue

                _, _, w, h = clipped
                if min(w, h) < self.config.min_face_size:
                    continue

                score = float(detection.score[0]) if detection.score else 0.0
                detections.append((clipped, score))

            return detections

        # Fallback backend: OpenCV Haar cascade.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        face_boxes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.config.min_face_size, self.config.min_face_size),
        )

        for (x, y, w, h) in face_boxes:
            clipped = self._clip_box(int(x), int(y), int(x + w), int(y + h), frame_w, frame_h)
            if clipped is not None:
                detections.append((clipped, 1.0))

        return detections

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces on a BGR frame.

        Returns detections in original frame coordinates.
        """
        processing_frame, scale = self._prepare_processing_frame(frame_bgr)
        raw = self._detect_raw(processing_frame)
        tracked = self._smoother.update(raw)

        if scale == 1.0:
            return tracked

        inv_scale = 1.0 / scale
        scaled_results: List[DetectedFace] = []

        for face in tracked:
            x, y, w, h = face.bbox
            scaled_results.append(
                DetectedFace(
                    track_id=face.track_id,
                    bbox=(
                        int(round(x * inv_scale)),
                        int(round(y * inv_scale)),
                        int(round(w * inv_scale)),
                        int(round(h * inv_scale)),
                    ),
                    score=face.score,
                )
            )

        return scaled_results


def draw_face_detections(frame_bgr: np.ndarray, faces: List[DetectedFace], show_track_id: bool = True) -> None:
    """Draw detection boxes and labels on the frame."""
    for face in faces:
        x, y, w, h = face.bbox
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 230, 120), 2)

        label = f"Face {face.score:.2f}"
        if show_track_id:
            label = f"ID {face.track_id} | {face.score:.2f}"

        label_y = max(y - 10, 15)
        cv2.putText(
            frame_bgr,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 230, 120),
            2,
            cv2.LINE_AA,
        )


def crop_face_regions(frame_bgr: np.ndarray, faces: List[DetectedFace]) -> List[Tuple[DetectedFace, np.ndarray]]:
    """Return cropped face images for downstream recognition pipelines."""
    crops: List[Tuple[DetectedFace, np.ndarray]] = []
    frame_h, frame_w = frame_bgr.shape[:2]

    for face in faces:
        x, y, w, h = face.bbox
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        crops.append((face, frame_bgr[y1:y2, x1:x2].copy()))

    return crops

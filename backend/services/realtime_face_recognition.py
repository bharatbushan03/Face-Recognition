from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

import cv2
import numpy as np

from backend.services.face_encoding_store import KnownFaceStore
from backend.services.realtime_face_detection import DetectedFace, MediaPipeFaceDetector


logger = logging.getLogger(__name__)


@dataclass
class RecognitionConfig:
    tolerance: float = 0.48
    ambiguity_margin: float = 0.03
    detect_every_n_frames: int = 1
    process_every_n_frames: int = 2
    encode_model: str = "small"
    max_faces_per_frame: int = 6
    encoding_num_workers: int = 1
    enable_low_light_enhancement: bool = True
    low_light_threshold: float = 70.0
    min_face_brightness: float = 35.0
    min_face_sharpness: float = 12.0


@dataclass(frozen=True)
class RecognizedFace:
    track_id: int
    bbox: tuple[int, int, int, int]
    name: str
    confidence: float
    distance: float
    status: str  # matched | unknown | ambiguous | no_encoding | low_quality


class RealtimeFaceRecognizer:
    """Runs real-time student recognition on detected faces."""

    def __init__(
        self,
        known_faces: KnownFaceStore,
        detector: MediaPipeFaceDetector,
        config: RecognitionConfig | None = None,
    ):
        self.known_faces = known_faces
        self.detector = detector
        self.config = config or RecognitionConfig()

        if self.config.process_every_n_frames <= 0:
            raise ValueError("process_every_n_frames must be > 0")
        if self.config.detect_every_n_frames <= 0:
            raise ValueError("detect_every_n_frames must be > 0")
        if self.config.max_faces_per_frame <= 0:
            raise ValueError("max_faces_per_frame must be > 0")
        if self.config.encoding_num_workers <= 0:
            raise ValueError("encoding_num_workers must be > 0")

        self._frame_index = 0
        self._last_detections: list[DetectedFace] = []
        self._last_results_by_track: dict[int, RecognizedFace] = {}
        self._face_recognition = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _get_face_recognition_module(self):
        if self._face_recognition is None:
            try:
                import face_recognition
            except ImportError as exc:
                raise ImportError(
                    "face_recognition is not installed. Run: python -m pip install face_recognition"
                ) from exc
            self._face_recognition = face_recognition
        return self._face_recognition

    @staticmethod
    def _estimate_frame_brightness(frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _enhance_low_light_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = self._clahe.apply(l_channel)
        merged = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _is_face_quality_acceptable(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        x, y, w, h = bbox
        frame_h, frame_w = frame_bgr.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)
        if x2 <= x1 or y2 <= y1:
            return False

        face_crop = frame_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            return False

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return brightness >= self.config.min_face_brightness and sharpness >= self.config.min_face_sharpness

    def _encode_single_face(self, frame_rgb: np.ndarray, location: tuple[int, int, int, int]) -> np.ndarray | None:
        face_recognition = self._get_face_recognition_module()
        encoded = face_recognition.face_encodings(
            frame_rgb,
            known_face_locations=[location],
            model=self.config.encode_model,
        )
        if not encoded:
            return None
        return np.asarray(encoded[0], dtype=np.float32)

    def _extract_face_encodings(
        self,
        frame_bgr: np.ndarray,
        detections: list[DetectedFace],
    ) -> tuple[list[np.ndarray | None], set[int]]:
        frame_for_encoding = frame_bgr
        if self.config.enable_low_light_enhancement:
            brightness = self._estimate_frame_brightness(frame_bgr)
            if brightness < self.config.low_light_threshold:
                frame_for_encoding = self._enhance_low_light_frame(frame_bgr)

        frame_h, frame_w = frame_for_encoding.shape[:2]
        frame_rgb = cv2.cvtColor(frame_for_encoding, cv2.COLOR_BGR2RGB)

        encodings: list[np.ndarray | None] = [None] * len(detections)
        low_quality_indices: set[int] = set()
        locations_for_batch: list[tuple[int, int, int, int]] = []
        mapping: list[int] = []

        for index, detection in enumerate(detections):
            if not self._is_face_quality_acceptable(frame_for_encoding, detection.bbox):
                low_quality_indices.add(index)
                continue

            location = self._bbox_to_face_location(
                detection.bbox,
                frame_width=frame_w,
                frame_height=frame_h,
            )
            locations_for_batch.append(location)
            mapping.append(index)

        if not locations_for_batch:
            return encodings, low_quality_indices

        face_recognition = self._get_face_recognition_module()
        batch_encodings = face_recognition.face_encodings(
            frame_rgb,
            known_face_locations=locations_for_batch,
            model=self.config.encode_model,
        )

        if len(batch_encodings) == len(locations_for_batch):
            for idx, encoded in enumerate(batch_encodings):
                target_index = mapping[idx]
                encodings[target_index] = np.asarray(encoded, dtype=np.float32)
            return encodings, low_quality_indices

        # Fallback path for rare cases where the library returns fewer encodings than locations.
        logger.warning(
            "Batch encoding count mismatch (locations=%s encodings=%s). Falling back to per-face encoding.",
            len(locations_for_batch),
            len(batch_encodings),
        )

        payload = list(enumerate(locations_for_batch))
        if self.config.encoding_num_workers > 1 and len(payload) > 1:
            with ThreadPoolExecutor(max_workers=self.config.encoding_num_workers) as executor:
                for batch_idx, encoded in executor.map(
                    lambda item: (item[0], self._encode_single_face(frame_rgb, item[1])),
                    payload,
                ):
                    target_index = mapping[batch_idx]
                    encodings[target_index] = encoded
        else:
            for batch_idx, location in payload:
                target_index = mapping[batch_idx]
                encodings[target_index] = self._encode_single_face(frame_rgb, location)

        return encodings, low_quality_indices

    @staticmethod
    def _bbox_to_face_location(
        bbox: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        x, y, w, h = bbox

        left = max(0, x)
        top = max(0, y)
        right = min(frame_width - 1, x + w)
        bottom = min(frame_height - 1, y + h)

        # face_recognition location format: (top, right, bottom, left)
        return top, right, bottom, left

    def _recognize_from_encodings(
        self,
        detections: list[DetectedFace],
        encodings: list[np.ndarray | None],
        low_quality_indices: set[int] | None = None,
    ) -> list[RecognizedFace]:
        low_quality_indices = low_quality_indices or set()
        results: list[RecognizedFace] = []

        valid_indices = [idx for idx, encoding in enumerate(encodings) if encoding is not None]
        valid_query_encodings: list[np.ndarray] = [encodings[idx] for idx in valid_indices if encodings[idx] is not None]
        batched_matches = self.known_faces.match_batch(
            query_encodings=valid_query_encodings,
            tolerance=self.config.tolerance,
            ambiguity_margin=self.config.ambiguity_margin,
        )
        match_by_index: dict[int, object] = {
            valid_indices[pos]: batched_matches[pos]
            for pos in range(len(valid_indices))
        }

        for index, detection in enumerate(detections):
            match = match_by_index.get(index)

            if match is None:
                status = "low_quality" if index in low_quality_indices else "no_encoding"
                result = RecognizedFace(
                    track_id=detection.track_id,
                    bbox=detection.bbox,
                    name="Unknown",
                    confidence=0.0,
                    distance=1.0,
                    status=status,
                )
            else:
                if match.is_match and match.name:
                    result = RecognizedFace(
                        track_id=detection.track_id,
                        bbox=detection.bbox,
                        name=match.name,
                        confidence=match.confidence,
                        distance=match.distance,
                        status="matched",
                    )
                elif match.is_ambiguous:
                    result = RecognizedFace(
                        track_id=detection.track_id,
                        bbox=detection.bbox,
                        name="Unknown",
                        confidence=match.confidence,
                        distance=match.distance,
                        status="ambiguous",
                    )
                else:
                    result = RecognizedFace(
                        track_id=detection.track_id,
                        bbox=detection.bbox,
                        name="Unknown",
                        confidence=match.confidence,
                        distance=match.distance,
                        status="unknown",
                    )

            self._last_results_by_track[detection.track_id] = result
            results.append(result)

        return results

    def _reuse_cached_results(self, detections: list[DetectedFace]) -> list[RecognizedFace]:
        cached_results: list[RecognizedFace] = []
        for detection in detections:
            cached = self._last_results_by_track.get(detection.track_id)
            if cached is None:
                cached_results.append(
                    RecognizedFace(
                        track_id=detection.track_id,
                        bbox=detection.bbox,
                        name="Unknown",
                        confidence=0.0,
                        distance=1.0,
                        status="unknown",
                    )
                )
            else:
                cached_results.append(
                    RecognizedFace(
                        track_id=detection.track_id,
                        bbox=detection.bbox,
                        name=cached.name,
                        confidence=cached.confidence,
                        distance=cached.distance,
                        status=cached.status,
                    )
                )
        return cached_results

    def _get_detections(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        should_detect = (
            not self._last_detections
            or (self._frame_index % self.config.detect_every_n_frames) == 0
        )
        if not should_detect:
            return list(self._last_detections)

        detections = self.detector.detect(frame_bgr)
        if len(detections) > self.config.max_faces_per_frame:
            detections = sorted(detections, key=lambda item: item.score, reverse=True)[: self.config.max_faces_per_frame]

        self._last_detections = list(detections)
        return detections

    def recognize_frame(self, frame_bgr: np.ndarray) -> list[RecognizedFace]:
        """Recognize students from a live BGR frame."""
        self._frame_index += 1

        detections = self._get_detections(frame_bgr)
        if not detections:
            self._last_detections = []
            return []

        should_encode = (self._frame_index % self.config.process_every_n_frames) == 0
        if not should_encode:
            # Reuse last labels between encoding frames for smoother UX and higher FPS.
            return self._reuse_cached_results(detections)

        encodings, low_quality_indices = self._extract_face_encodings(frame_bgr, detections)
        return self._recognize_from_encodings(
            detections=detections,
            encodings=encodings,
            low_quality_indices=low_quality_indices,
        )


def draw_recognition_results(frame_bgr: np.ndarray, results: list[RecognizedFace]) -> None:
    """Draw recognition labels and boxes on the frame."""
    for result in results:
        x, y, w, h = result.bbox

        if result.status == "matched":
            color = (30, 220, 80)
        elif result.status == "ambiguous":
            color = (0, 190, 255)
        elif result.status == "low_quality":
            color = (0, 210, 255)
        else:
            color = (0, 90, 255)

        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)

        label = result.name
        if result.status == "matched":
            label = f"{result.name} {result.confidence:.2f}"
        elif result.status == "ambiguous":
            label = f"Unknown (ambiguous {result.confidence:.2f})"
        elif result.status == "low_quality":
            label = "Unknown (low quality)"

        label_y = max(y - 10, 15)
        cv2.putText(
            frame_bgr,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

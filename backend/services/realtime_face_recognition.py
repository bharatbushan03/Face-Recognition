from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.services.face_encoding_store import KnownFaceStore
from backend.services.realtime_face_detection import DetectedFace, MediaPipeFaceDetector


@dataclass
class RecognitionConfig:
    tolerance: float = 0.48
    ambiguity_margin: float = 0.03
    process_every_n_frames: int = 2
    encode_model: str = "small"


@dataclass(frozen=True)
class RecognizedFace:
    track_id: int
    bbox: tuple[int, int, int, int]
    name: str
    confidence: float
    distance: float
    status: str  # matched | unknown | ambiguous | no_encoding


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

        self._frame_index = 0
        self._last_results_by_track: dict[int, RecognizedFace] = {}

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
    ) -> list[RecognizedFace]:
        results: list[RecognizedFace] = []

        for index, detection in enumerate(detections):
            face_encoding = encodings[index] if index < len(encodings) else None

            if face_encoding is None:
                result = RecognizedFace(
                    track_id=detection.track_id,
                    bbox=detection.bbox,
                    name="Unknown",
                    confidence=0.0,
                    distance=1.0,
                    status="no_encoding",
                )
            else:
                match = self.known_faces.match(
                    query_encoding=face_encoding,
                    tolerance=self.config.tolerance,
                    ambiguity_margin=self.config.ambiguity_margin,
                )

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

    def recognize_frame(self, frame_bgr: np.ndarray) -> list[RecognizedFace]:
        """Recognize students from a live BGR frame."""
        self._frame_index += 1

        detections = self.detector.detect(frame_bgr)
        if not detections:
            return []

        should_encode = (self._frame_index % self.config.process_every_n_frames) == 0
        if not should_encode:
            # Reuse last labels between encoding frames for smoother UX and higher FPS.
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

        try:
            import face_recognition
        except ImportError as exc:
            raise ImportError(
                "face_recognition is not installed. Run: python -m pip install face_recognition"
            ) from exc

        frame_h, frame_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face_locations = [
            self._bbox_to_face_location(face.bbox, frame_width=frame_w, frame_height=frame_h)
            for face in detections
        ]

        encodings: list[np.ndarray] = []
        for location in face_locations:
            encoded = face_recognition.face_encodings(
                frame_rgb,
                known_face_locations=[location],
                model=self.config.encode_model,
            )
            encodings.append(encoded[0] if encoded else None)

        return self._recognize_from_encodings(detections=detections, encodings=encodings)


def draw_recognition_results(frame_bgr: np.ndarray, results: list[RecognizedFace]) -> None:
    """Draw recognition labels and boxes on the frame."""
    for result in results:
        x, y, w, h = result.bbox

        if result.status == "matched":
            color = (30, 220, 80)
        elif result.status == "ambiguous":
            color = (0, 190, 255)
        else:
            color = (0, 90, 255)

        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)

        label = result.name
        if result.status == "matched":
            label = f"{result.name} {result.confidence:.2f}"
        elif result.status == "ambiguous":
            label = f"Unknown (ambiguous {result.confidence:.2f})"

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

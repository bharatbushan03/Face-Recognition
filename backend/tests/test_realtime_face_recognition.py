import numpy as np
import pytest

from backend.services.face_encoding_store import KnownFaceStore
from backend.services.realtime_face_detection import DetectedFace
from backend.services.realtime_face_recognition import RecognitionConfig, RealtimeFaceRecognizer


class _FakeDetector:
    def __init__(self, detections: list[DetectedFace]):
        self._detections = list(detections)
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        self.calls += 1
        return list(self._detections)


def test_detect_every_n_frames_reuses_last_detections():
    detector = _FakeDetector(
        [
            DetectedFace(track_id=7, bbox=(4, 5, 20, 20), score=0.93),
        ]
    )
    recognizer = RealtimeFaceRecognizer(
        known_faces=KnownFaceStore.empty(),
        detector=detector,
        config=RecognitionConfig(
            detect_every_n_frames=2,
            process_every_n_frames=10,
        ),
    )

    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    first = recognizer.recognize_frame(frame)
    second = recognizer.recognize_frame(frame)
    third = recognizer.recognize_frame(frame)

    assert detector.calls == 2
    assert len(first) == 1
    assert len(second) == 1
    assert len(third) == 1
    assert third[0].bbox == (4, 5, 20, 20)
    assert all(item.status == "unknown" for item in first + second + third)


def test_detect_every_n_frames_must_be_positive():
    detector = _FakeDetector([])

    with pytest.raises(ValueError, match="detect_every_n_frames must be > 0"):
        RealtimeFaceRecognizer(
            known_faces=KnownFaceStore.empty(),
            detector=detector,
            config=RecognitionConfig(detect_every_n_frames=0),
        )

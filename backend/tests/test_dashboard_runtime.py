import numpy as np

from backend.services.face_encoding_store import KnownFaceStore
from backend.services.realtime_face_detection import DetectedFace
from backend.services.realtime_face_recognition import RecognizedFace
from backend.ui import dashboard_runtime as dashboard_runtime_module
from backend.ui.dashboard_runtime import AttendanceDashboardRuntime, DashboardRuntimeConfig


class _FakeCapture:
    def __init__(self, frame: np.ndarray):
        self._frame = frame
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self._frame.copy()

    def release(self) -> None:
        self._opened = False


class _FakeRecognizer:
    def __init__(self, result: RecognizedFace):
        self.result = result
        self.frames_seen: list[np.ndarray] = []

    def recognize_frame(self, frame_bgr: np.ndarray) -> list[RecognizedFace]:
        self.frames_seen.append(frame_bgr.copy())
        return [self.result]


class _FakeDetector:
    def __init__(self, detections: list[DetectedFace]):
        self._detections = list(detections)
        self.calls = 0
        self.backend_name = "fake_detector"

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        self.calls += 1
        return list(self._detections)


class _CompatRecognitionConfig:
    def __init__(
        self,
        tolerance=0.48,
        ambiguity_margin=0.03,
        process_every_n_frames=2,
        encode_model="small",
        max_faces_per_frame=6,
        enable_low_light_enhancement=True,
        low_light_threshold=70.0,
        min_face_brightness=35.0,
        min_face_sharpness=12.0,
    ):
        self.tolerance = tolerance
        self.ambiguity_margin = ambiguity_margin
        self.process_every_n_frames = process_every_n_frames
        self.encode_model = encode_model
        self.max_faces_per_frame = max_faces_per_frame
        self.enable_low_light_enhancement = enable_low_light_enhancement
        self.low_light_threshold = low_light_threshold
        self.min_face_brightness = min_face_brightness
        self.min_face_sharpness = min_face_sharpness


class _FakeRuntimeRecognizer:
    def __init__(self, known_faces, detector, config):
        self.known_faces = known_faces
        self.detector = detector
        self.config = config


def _build_runtime(tmp_path, monkeypatch) -> AttendanceDashboardRuntime:
    monkeypatch.setattr(
        AttendanceDashboardRuntime,
        "_load_or_build_known_faces",
        lambda self, refresh_cache=None: KnownFaceStore.empty(),
    )

    return AttendanceDashboardRuntime(
        DashboardRuntimeConfig(
            dataset_dir=str(tmp_path / "students"),
            cache_file=str(tmp_path / "known_faces.npz"),
            attendance_file=str(tmp_path / "attendance.csv"),
            mirror=False,
            display_smoothing_alpha=0.25,
            frame_width=4,
            frame_height=4,
        )
    )


def test_process_next_frame_smooths_display_without_blurring_recognition(tmp_path, monkeypatch):
    runtime = _build_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(dashboard_runtime_module, "draw_recognition_results", lambda frame, results: None)

    current_frame = np.full((4, 4, 3), 100, dtype=np.uint8)
    previous_frame = np.full((4, 4, 3), 200, dtype=np.uint8)

    fake_recognizer = _FakeRecognizer(
        RecognizedFace(
            track_id=1,
            bbox=(0, 0, 1, 1),
            name="Unknown",
            confidence=0.0,
            distance=1.0,
            status="unknown",
        )
    )

    runtime._known_faces = KnownFaceStore(
        names=["Alice"],
        encodings=np.zeros((1, 128), dtype=np.float32),
    )
    runtime.recognizer = fake_recognizer
    runtime._cap = _FakeCapture(current_frame)
    runtime._last_display_bgr = previous_frame.copy()

    update = runtime.process_next_frame()

    assert len(fake_recognizer.frames_seen) == 1
    np.testing.assert_array_equal(fake_recognizer.frames_seen[0], current_frame)
    assert int(update.frame_rgb[0, 0, 0]) == 125
    assert int(runtime._last_display_bgr[0, 0, 0]) == 125


def test_stop_camera_clears_cached_frame_state(tmp_path, monkeypatch):
    runtime = _build_runtime(tmp_path, monkeypatch)
    runtime._cap = _FakeCapture(np.zeros((2, 2, 3), dtype=np.uint8))
    runtime._last_frame_rgb = np.ones((2, 2, 3), dtype=np.uint8)
    runtime._last_display_bgr = np.ones((2, 2, 3), dtype=np.uint8)
    runtime._consecutive_read_failures = 4

    runtime.stop_camera()

    assert runtime._cap is None
    assert runtime._last_frame_rgb is None
    assert runtime._last_display_bgr is None
    assert runtime._consecutive_read_failures == 0


def test_process_next_frame_reuses_unknown_detections_between_detection_passes(tmp_path, monkeypatch):
    runtime = _build_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(dashboard_runtime_module, "draw_recognition_results", lambda frame, results: None)

    runtime.detector = _FakeDetector(
        [
            DetectedFace(track_id=3, bbox=(1, 2, 8, 8), score=0.88),
        ]
    )
    runtime._cap = _FakeCapture(np.zeros((16, 16, 3), dtype=np.uint8))

    first = runtime.process_next_frame()
    second = runtime.process_next_frame()
    third = runtime.process_next_frame()

    assert runtime.detector.calls == 2
    assert first.total_faces == 1
    assert second.total_faces == 1
    assert third.total_faces == 1


def test_runtime_handles_older_recognition_config_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_runtime_module, "RecognitionConfig", _CompatRecognitionConfig)
    monkeypatch.setattr(dashboard_runtime_module, "RealtimeFaceRecognizer", _FakeRuntimeRecognizer)

    runtime = _build_runtime(tmp_path, monkeypatch)

    assert isinstance(runtime._recognition_config, _CompatRecognitionConfig)
    assert runtime.recognizer.config.process_every_n_frames == 2

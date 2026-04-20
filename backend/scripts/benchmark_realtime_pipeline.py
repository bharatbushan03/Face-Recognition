from __future__ import annotations

import argparse
from pathlib import Path
import platform
import time
import logging

import cv2
import numpy as np

from backend.services.face_encoding_store import KnownFaceStore, build_known_face_store
from backend.services.realtime_face_detection import FaceDetectionConfig, MediaPipeFaceDetector
from backend.services.realtime_face_recognition import RecognitionConfig, RealtimeFaceRecognizer
from backend.utils.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


def _open_capture(camera_index: int, video_path: str | None, camera_buffer_size: int = 1) -> cv2.VideoCapture:
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        return cap

    if platform.system().lower() == "windows":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            if camera_buffer_size > 0:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, camera_buffer_size)
            return cap

    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened() and camera_buffer_size > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, camera_buffer_size)
    return cap


def _load_known_faces(dataset_dir: Path, cache_path: Path, refresh_cache: bool) -> KnownFaceStore:
    if cache_path.exists() and not refresh_cache:
        try:
            store = KnownFaceStore.load_npz(cache_path)
            if store.size > 0:
                return store
        except Exception as exc:
            logger.warning("Unable to load cache %s (%s). Rebuilding.", cache_path, exc)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    store, report = build_known_face_store(dataset_dir=dataset_dir, detection_model="hog", num_jitters=1)
    store.save_npz(cache_path)
    logger.info(
        "Built cache for benchmark | students=%s images=%s encodings=%s",
        report.students_found,
        report.images_scanned,
        report.encodings_created,
    )
    return store


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def run_benchmark(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    cache_path = Path(args.cache_file)

    known_faces = _load_known_faces(dataset_dir=dataset_dir, cache_path=cache_path, refresh_cache=args.refresh_cache)
    if known_faces.size == 0:
        raise RuntimeError("No known face encodings found. Add student images first.")

    detector = MediaPipeFaceDetector(
        FaceDetectionConfig(
            process_width=args.detect_process_width,
            min_detection_confidence=args.detect_min_confidence,
            model_selection=0,
            smoothing_alpha=0.65,
            max_tracking_distance=90.0,
            max_missing_frames=5,
            min_face_size=args.min_face_size,
        )
    )

    recognizer = RealtimeFaceRecognizer(
        known_faces=known_faces,
        detector=detector,
        config=RecognitionConfig(
            tolerance=args.tolerance,
            ambiguity_margin=args.ambiguity_margin,
            detect_every_n_frames=args.detect_every_n_frames,
            process_every_n_frames=args.process_every_n_frames,
            encode_model=args.encode_model,
            max_faces_per_frame=args.max_faces_per_frame,
            encoding_num_workers=args.encoding_num_workers,
            enable_low_light_enhancement=not args.disable_low_light_enhancement,
            low_light_threshold=args.low_light_threshold,
            min_face_brightness=args.min_face_brightness,
            min_face_sharpness=args.min_face_sharpness,
        ),
    )

    cap = _open_capture(
        camera_index=args.camera_index,
        video_path=args.video_file,
        camera_buffer_size=args.camera_buffer_size,
    )
    if not cap.isOpened():
        raise RuntimeError("Could not open capture source.")

    latencies_ms: list[float] = []
    total_faces = 0
    matched_faces = 0
    processed_frames = 0
    started_at = time.perf_counter()

    try:
        while processed_frames < args.frames:
            ok, frame = cap.read()
            if not ok:
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            tic = time.perf_counter()
            results = recognizer.recognize_frame(frame)
            toc = time.perf_counter()

            processed_frames += 1
            latencies_ms.append((toc - tic) * 1000.0)

            total_faces += len(results)
            matched_faces += sum(1 for item in results if item.status == "matched")

    finally:
        cap.release()

    total_runtime_s = max(time.perf_counter() - started_at, 1e-6)
    fps = processed_frames / total_runtime_s

    mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    p50_latency = _percentile(latencies_ms, 50)
    p95_latency = _percentile(latencies_ms, 95)

    print("\nRealtime Pipeline Benchmark")
    print("-" * 50)
    print(f"Frames processed     : {processed_frames}")
    print(f"Runtime (s)          : {total_runtime_s:.2f}")
    print(f"Effective FPS        : {fps:.2f}")
    print(f"Mean latency (ms)    : {mean_latency:.2f}")
    print(f"P50 latency (ms)     : {p50_latency:.2f}")
    print(f"P95 latency (ms)     : {p95_latency:.2f}")
    print(f"Total faces observed : {total_faces}")
    print(f"Matched faces        : {matched_faces}")
    print(f"Detection backend    : {detector.backend_name}")
    print("-" * 50)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark realtime face recognition pipeline")

    parser.add_argument("--dataset-dir", type=str, default="data/students")
    parser.add_argument("--cache-file", type=str, default="models/known_face_encodings.npz")
    parser.add_argument("--refresh-cache", action="store_true")

    parser.add_argument("--frames", type=int, default=300, help="Number of frames to process")
    parser.add_argument("--video-file", type=str, default=None, help="Optional video file path")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-buffer-size", type=int, default=1)
    parser.add_argument("--mirror", action="store_true")

    parser.add_argument("--detect-process-width", type=int, default=640)
    parser.add_argument("--detect-min-confidence", type=float, default=0.65)
    parser.add_argument("--min-face-size", type=int, default=28)

    parser.add_argument("--tolerance", type=float, default=0.48)
    parser.add_argument("--ambiguity-margin", type=float, default=0.03)
    parser.add_argument("--detect-every-n-frames", type=int, default=2)
    parser.add_argument("--process-every-n-frames", type=int, default=2)
    parser.add_argument("--encode-model", choices=["small", "large"], default="small")
    parser.add_argument("--max-faces-per-frame", type=int, default=6)
    parser.add_argument("--encoding-num-workers", type=int, default=1)
    parser.add_argument("--disable-low-light-enhancement", action="store_true")
    parser.add_argument("--low-light-threshold", type=float, default=70.0)
    parser.add_argument("--min-face-brightness", type=float, default=35.0)
    parser.add_argument("--min-face-sharpness", type=float, default=12.0)

    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())

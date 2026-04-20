import argparse
import platform
import time
import logging

import cv2

from backend.services.realtime_face_detection import (
    FaceDetectionConfig,
    MediaPipeFaceDetector,
    draw_face_detections,
)
from backend.utils.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


def _ensure_opencv_gui() -> None:
    """Fail fast when OpenCV GUI support is unavailable (headless install)."""
    window_name = "opencv_gui_check"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(window_name)
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV GUI is unavailable. Install desktop OpenCV with: "
            "python -m pip install opencv-python"
        ) from exc


def _open_camera(camera_index: int, camera_buffer_size: int = 1) -> cv2.VideoCapture:
    """Open webcam with Windows-friendly backend fallback."""
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


def run_realtime_detection(args: argparse.Namespace) -> None:
    _ensure_opencv_gui()

    config = FaceDetectionConfig(
        process_width=args.process_width,
        min_detection_confidence=args.min_confidence,
        model_selection=args.model_selection,
        smoothing_alpha=args.smoothing_alpha,
        max_tracking_distance=args.max_tracking_distance,
        max_missing_frames=args.max_missing_frames,
        min_face_size=args.min_face_size,
    )
    detector = MediaPipeFaceDetector(config=config)
    logger.info("Using detection backend: %s", detector.backend_name)

    cap = _open_camera(args.camera_index, camera_buffer_size=args.camera_buffer_size)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not access webcam. Check permissions, close other camera apps, or try --camera-index 1"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)

    logger.info("Starting real-time face detection. Press 'q' to quit safely.")

    prev_time = time.perf_counter()
    smoothed_fps = 0.0
    consecutive_read_failures = 0
    frame_counter = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_read_failures += 1
                logger.warning(
                    "Failed to read webcam frame (%s/%s)",
                    consecutive_read_failures,
                    args.max_read_failures,
                )
                if consecutive_read_failures >= args.max_read_failures:
                    logger.error("Exceeded max camera read failures. Exiting loop.")
                    break
                continue

            consecutive_read_failures = 0
            frame_counter += 1

            if args.mirror:
                frame = cv2.flip(frame, 1)

            faces = detector.detect(frame)
            draw_face_detections(frame, faces, show_track_id=True)

            if frame_counter % 150 == 0:
                logger.info("Runtime stats | Faces=%s FPS=%.1f", len(faces), smoothed_fps)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            smoothed_fps = (0.9 * smoothed_fps) + (0.1 * instant_fps) if smoothed_fps > 0 else instant_fps

            cv2.putText(
                frame,
                f"FPS: {smoothed_fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Backend: {detector.backend_name}",
                (15, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press 'q' to exit",
                (15, 126),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 220, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Face Detection - Attendance Prototype", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time webcam face detection")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--camera-buffer-size",
        type=int,
        default=1,
        help="OpenCV camera buffer size. Lower values reduce webcam latency.",
    )
    parser.add_argument("--frame-width", type=int, default=1280, help="Webcam capture width")
    parser.add_argument("--frame-height", type=int, default=720, help="Webcam capture height")
    parser.add_argument(
        "--process-width",
        type=int,
        default=640,
        help="Frame width for detection inference (lower = faster, less precise)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum confidence threshold for face detections",
    )
    parser.add_argument(
        "--model-selection",
        type=int,
        choices=[0, 1],
        default=0,
        help="MediaPipe model type (0=short range, 1=long range)",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.65,
        help="Bounding box smoothing factor. Higher = smoother but less reactive",
    )
    parser.add_argument(
        "--max-tracking-distance",
        type=float,
        default=90.0,
        help="Max center distance (pixels in processed frame) to keep same track ID",
    )
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=5,
        help="How many missed frames before dropping a track",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=28,
        help="Minimum face size in pixels",
    )
    parser.add_argument(
        "--max-read-failures",
        type=int,
        default=30,
        help="Maximum consecutive camera read failures before stopping",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror display output for easier classroom operator feedback",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_realtime_detection(parse_args())

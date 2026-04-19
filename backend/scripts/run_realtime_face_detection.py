import argparse
import platform
import time

import cv2

from backend.services.realtime_face_detection import (
    FaceDetectionConfig,
    MediaPipeFaceDetector,
    draw_face_detections,
)


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


def _open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open webcam with Windows-friendly backend fallback."""
    if platform.system().lower() == "windows":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap

    return cv2.VideoCapture(camera_index)


def run_realtime_detection(args: argparse.Namespace) -> None:
    _ensure_opencv_gui()

    config = FaceDetectionConfig(
        process_width=args.process_width,
        min_detection_confidence=args.min_confidence,
        model_selection=args.model_selection,
        smoothing_alpha=args.smoothing_alpha,
        max_tracking_distance=args.max_tracking_distance,
        max_missing_frames=args.max_missing_frames,
    )
    detector = MediaPipeFaceDetector(config=config)
    print(f"Using detection backend: {detector.backend_name}")

    cap = _open_camera(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not access webcam. Check permissions, close other camera apps, or try --camera-index 1"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)

    print("Starting real-time face detection...")
    print("Press 'q' to quit safely.")

    prev_time = time.perf_counter()
    smoothed_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read webcam frame. Exiting loop.")
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            faces = detector.detect(frame)
            draw_face_detections(frame, faces, show_track_id=True)

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
        "--mirror",
        action="store_true",
        help="Mirror display output for easier classroom operator feedback",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_realtime_detection(parse_args())

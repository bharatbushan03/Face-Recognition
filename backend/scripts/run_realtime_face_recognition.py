from __future__ import annotations

import argparse
from pathlib import Path
import platform
import time

import cv2

from backend.services.attendance_service import AttendanceConfig, AttendanceManager
from backend.services.face_encoding_store import KnownFaceStore, build_known_face_store
from backend.services.realtime_face_detection import FaceDetectionConfig, MediaPipeFaceDetector
from backend.services.realtime_face_recognition import (
    RecognitionConfig,
    RealtimeFaceRecognizer,
    draw_recognition_results,
)


def _ensure_opencv_gui() -> None:
    """Fail fast if OpenCV GUI support is unavailable."""
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
    if platform.system().lower() == "windows":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(camera_index)


def _load_or_build_known_faces(
    dataset_dir: Path,
    cache_path: Path,
    refresh_cache: bool,
    build_model: str,
    num_jitters: int,
) -> KnownFaceStore:
    if cache_path.exists() and not refresh_cache:
        known_faces = KnownFaceStore.load_npz(cache_path)
        print(f"Loaded {known_faces.size} known face encodings from cache: {cache_path}")
        return known_faces

    known_faces, report = build_known_face_store(
        dataset_dir=dataset_dir,
        detection_model=build_model,
        num_jitters=num_jitters,
    )
    known_faces.save_npz(cache_path)

    print("Built known face encodings from dataset")
    print(
        f"Students: {report.students_found} | Images scanned: {report.images_scanned} | "
        f"Encodings: {report.encodings_created} | Skipped(no-face): {report.skipped_no_face} | "
        f"Skipped(errors): {report.skipped_errors}"
    )
    print(f"Saved encoding cache to: {cache_path}")
    return known_faces


def run_realtime_recognition(args: argparse.Namespace) -> None:
    _ensure_opencv_gui()

    dataset_dir = Path(args.dataset_dir)
    cache_path = Path(args.cache_file)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            "Create it and add student images first."
        )

    known_faces = _load_or_build_known_faces(
        dataset_dir=dataset_dir,
        cache_path=cache_path,
        refresh_cache=args.refresh_cache,
        build_model=args.build_model,
        num_jitters=args.num_jitters,
    )
    if known_faces.size == 0:
        raise RuntimeError("No known encodings were created. Check your student images and try again.")

    detector = MediaPipeFaceDetector(
        FaceDetectionConfig(
            process_width=args.detect_process_width,
            min_detection_confidence=args.detect_min_confidence,
            model_selection=args.detect_model_selection,
            smoothing_alpha=args.detect_smoothing_alpha,
            max_tracking_distance=args.detect_max_tracking_distance,
            max_missing_frames=args.detect_max_missing_frames,
            min_face_size=args.min_face_size,
        )
    )

    recognizer = RealtimeFaceRecognizer(
        known_faces=known_faces,
        detector=detector,
        config=RecognitionConfig(
            tolerance=args.tolerance,
            ambiguity_margin=args.ambiguity_margin,
            process_every_n_frames=args.process_every_n_frames,
            encode_model=args.encode_model,
        ),
    )

    attendance_manager = AttendanceManager(
        AttendanceConfig(
            csv_path=args.attendance_file,
            dedupe_scope=args.dedupe_scope,
            session_id=args.session_id,
        )
    )

    cap = _open_camera(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not access webcam. Check permissions, close other camera apps, or try --camera-index 1"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)

    print(f"Using detection backend: {detector.backend_name}")
    print(f"Known encodings loaded: {known_faces.size}")
    print(f"Attendance file: {Path(args.attendance_file)}")
    print(
        f"Attendance session started: {attendance_manager.session_id} "
        f"(dedupe_scope={args.dedupe_scope})"
    )
    print("Starting real-time face recognition...")
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

            results = recognizer.recognize_frame(frame)

            # Mark each recognized student at most once in this frame, then
            # rely on AttendanceManager duplicate protection for day/session scope.
            matched_in_frame: dict[str, float] = {}
            for item in results:
                if item.status != "matched":
                    continue

                prev_conf = matched_in_frame.get(item.name)
                if prev_conf is None or item.confidence > prev_conf:
                    matched_in_frame[item.name] = item.confidence

            for student_name, confidence in matched_in_frame.items():
                marked, record = attendance_manager.mark_present(
                    student_name=student_name,
                    confidence=confidence,
                    source="realtime_face_recognition",
                )
                if marked and record is not None:
                    print(f"Attendance marked for {record.student_name} at {record.time}")

            draw_recognition_results(frame, results)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            smoothed_fps = (0.9 * smoothed_fps) + (0.1 * instant_fps) if smoothed_fps > 0 else instant_fps

            known_count = sum(1 for item in results if item.status == "matched")
            unknown_count = len(results) - known_count

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
                f"Known: {known_count}  Unknown: {unknown_count}",
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

            cv2.imshow("Face Recognition - Attendance Prototype", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(
            f"Attendance session ended: {attendance_manager.session_id}. "
            f"Log file: {Path(args.attendance_file)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time webcam face recognition")

    # Dataset and cache
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/students",
        help="Directory containing student face images",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="models/known_face_encodings.npz",
        help="Path to save/load known encodings cache",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild encodings from dataset instead of using cache",
    )
    parser.add_argument(
        "--build-model",
        type=str,
        default="hog",
        choices=["hog", "cnn"],
        help="face_recognition model used while building known encodings",
    )
    parser.add_argument(
        "--num-jitters",
        type=int,
        default=1,
        help="face_recognition encoding jitters while building known encodings",
    )

    # Webcam settings
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index")
    parser.add_argument("--frame-width", type=int, default=1280, help="Webcam capture width")
    parser.add_argument("--frame-height", type=int, default=720, help="Webcam capture height")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror display output for easier classroom operator feedback",
    )

    # Detection settings
    parser.add_argument("--detect-process-width", type=int, default=640, help="Detection process width")
    parser.add_argument("--detect-min-confidence", type=float, default=0.65, help="Detection confidence")
    parser.add_argument(
        "--detect-model-selection",
        type=int,
        choices=[0, 1],
        default=0,
        help="MediaPipe detection model selection",
    )
    parser.add_argument("--detect-smoothing-alpha", type=float, default=0.65, help="Detection smoothing alpha")
    parser.add_argument(
        "--detect-max-tracking-distance",
        type=float,
        default=90.0,
        help="Tracking distance for stable face IDs",
    )
    parser.add_argument(
        "--detect-max-missing-frames",
        type=int,
        default=5,
        help="How long to keep a track alive when temporarily missing",
    )
    parser.add_argument("--min-face-size", type=int, default=28, help="Minimum face size in pixels")

    # Recognition settings
    parser.add_argument("--tolerance", type=float, default=0.48, help="Distance threshold for match decision")
    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=0.03,
        help="Distance gap for ambiguous top-2 candidates",
    )
    parser.add_argument(
        "--process-every-n-frames",
        type=int,
        default=2,
        help="Run face encoding every N frames to improve performance",
    )
    parser.add_argument(
        "--encode-model",
        type=str,
        choices=["small", "large"],
        default="small",
        help="face_recognition encoding model for live frames",
    )

    # Attendance settings
    parser.add_argument(
        "--attendance-file",
        type=str,
        default="attendance_logs/attendance.csv",
        help="CSV file path for attendance logs",
    )
    parser.add_argument(
        "--dedupe-scope",
        type=str,
        choices=["day", "session"],
        default="day",
        help="Duplicate prevention scope: per day or per session",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Optional explicit session ID. Defaults to timestamp when process starts.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_realtime_recognition(parse_args())

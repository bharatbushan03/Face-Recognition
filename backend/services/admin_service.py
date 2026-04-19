from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import re
import platform

import cv2

from backend.services.face_encoding_store import (
    EncodingBuildReport,
    KnownFaceStore,
    build_known_face_store,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class StudentInfo:
    folder_name: str
    student_id: str
    student_name: str
    image_count: int


@dataclass(frozen=True)
class AddStudentResult:
    success: bool
    message: str
    student: StudentInfo | None
    report: EncodingBuildReport | None
    skipped_images: list[str]


@dataclass(frozen=True)
class RemoveStudentResult:
    success: bool
    message: str
    removed_student: StudentInfo | None
    report: EncodingBuildReport | None


def _slugify(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_\-]", "", value)
    return re.sub(r"_+", "_", value).strip("_")


def _parse_student_folder(folder_name: str) -> tuple[str, str]:
    if "__" in folder_name:
        student_id, raw_name = folder_name.split("__", 1)
        student_name = raw_name.replace("_", " ").strip() or raw_name
        return student_id.strip(), student_name

    student_name = folder_name.replace("_", " ").strip() or folder_name
    return "", student_name


def _student_folder_name(student_name: str, student_id: str | None = None) -> str:
    safe_name = _slugify(student_name)
    if not safe_name:
        raise ValueError("Student name produced an empty folder name after sanitization")

    safe_id = _slugify(student_id) if student_id else ""
    if safe_id:
        return f"{safe_id}__{safe_name}"
    return safe_name


def list_students(dataset_dir: str | Path) -> list[StudentInfo]:
    path = Path(dataset_dir)
    if not path.exists():
        return []

    students: list[StudentInfo] = []
    for item in sorted(path.iterdir(), key=lambda p: p.name.lower()):
        if not item.is_dir():
            continue

        student_id, student_name = _parse_student_folder(item.name)
        image_count = sum(
            1
            for image in item.glob("*")
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
        )
        students.append(
            StudentInfo(
                folder_name=item.name,
                student_id=student_id,
                student_name=student_name,
                image_count=image_count,
            )
        )
    return students


def load_encodings(cache_path: str | Path) -> KnownFaceStore:
    path = Path(cache_path)
    if not path.exists():
        return KnownFaceStore.empty()

    try:
        return KnownFaceStore.load_npz(path)
    except Exception as exc:
        raise RuntimeError(
            f"Encoding cache is corrupted or unreadable at {path}. "
            "Rebuild encodings using the admin re-encode option."
        ) from exc


def save_encodings(store: KnownFaceStore, cache_path: str | Path) -> None:
    store.save_npz(cache_path)


def rebuild_and_save_encodings(
    dataset_dir: str | Path,
    cache_path: str | Path,
    detection_model: str = "hog",
    num_jitters: int = 1,
) -> EncodingBuildReport:
    known_store, report = build_known_face_store(
        dataset_dir=dataset_dir,
        detection_model=detection_model,
        num_jitters=num_jitters,
    )
    save_encodings(known_store, cache_path)
    return report


def _validate_single_face_image(image_path: Path) -> tuple[bool, str | None]:
    try:
        import face_recognition
    except ImportError as exc:
        raise ImportError(
            "face_recognition is not installed. Run: python -m pip install face_recognition"
        ) from exc

    try:
        image = face_recognition.load_image_file(str(image_path))
        locations = face_recognition.face_locations(image, model="hog")
    except Exception as exc:
        return False, f"Failed to process image: {exc}"

    if len(locations) == 0:
        return False, "No face detected"
    if len(locations) > 1:
        return False, "Multiple faces detected"
    return True, None


def _copy_images_to_student_folder(image_paths: list[str], student_folder: Path) -> list[Path]:
    copied: list[Path] = []
    next_index = 1

    for source_path in image_paths:
        src = Path(source_path)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Image file not found: {src}")
        if src.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {src}")

        while True:
            target = student_folder / f"img_{next_index:03d}{src.suffix.lower()}"
            next_index += 1
            if not target.exists():
                break

        shutil.copy2(src, target)
        copied.append(target)

    return copied


def _capture_images_from_webcam(
    student_folder: Path,
    count: int,
    camera_index: int = 0,
) -> list[Path]:
    if count <= 0:
        raise ValueError("Capture count must be > 0")

    if platform.system().lower() == "windows":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check camera permissions and index.")

    captured: list[Path] = []
    try:
        while len(captured) < count:
            ok, frame = cap.read()
            if not ok:
                continue

            preview = frame.copy()
            cv2.putText(
                preview,
                f"Capture {len(captured) + 1}/{count} | Press C to capture | Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (30, 220, 80),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Add Student - Webcam Capture", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c"):
                output_path = student_folder / f"capture_{len(captured) + 1:03d}.jpg"
                cv2.imwrite(str(output_path), frame)
                captured.append(output_path)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured


def _find_student_folder(dataset_path: Path, student_ref: str) -> Path | None:
    ref = student_ref.strip().lower()
    if not ref:
        return None

    candidates: list[Path] = []
    for student in list_students(dataset_path):
        if student.folder_name.lower() == ref:
            candidates.append(dataset_path / student.folder_name)
            continue
        if student.student_id and student.student_id.lower() == ref:
            candidates.append(dataset_path / student.folder_name)
            continue
        if student.student_name.lower() == ref:
            candidates.append(dataset_path / student.folder_name)

    if len(candidates) == 1:
        return candidates[0]
    return None


def add_student(
    student_name: str,
    student_id: str | None,
    dataset_dir: str | Path,
    cache_path: str | Path,
    image_paths: list[str] | None = None,
    capture_from_webcam: bool = False,
    capture_count: int = 5,
    camera_index: int = 0,
) -> AddStudentResult:
    name = student_name.strip()
    sid = (student_id or "").strip()
    if not name:
        return AddStudentResult(False, "Student name is required", None, None, [])

    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)

    folder_name = _student_folder_name(name, sid)
    student_folder = dataset_path / folder_name

    existing = list_students(dataset_path)
    if student_folder.exists():
        return AddStudentResult(
            False,
            f"Student already exists: {folder_name}",
            None,
            None,
            [],
        )

    if sid and any(item.student_id == sid for item in existing if item.student_id):
        return AddStudentResult(
            False,
            f"Duplicate student_id detected: {sid}",
            None,
            None,
            [],
        )

    if not sid and any(item.student_name.lower() == name.lower() for item in existing):
        return AddStudentResult(
            False,
            f"Duplicate student name detected: {name}",
            None,
            None,
            [],
        )

    if not image_paths and not capture_from_webcam:
        return AddStudentResult(
            False,
            "Provide image_paths or enable capture_from_webcam",
            None,
            None,
            [],
        )

    student_folder.mkdir(parents=True, exist_ok=False)
    skipped: list[str] = []

    try:
        imported: list[Path] = []
        if image_paths:
            imported.extend(_copy_images_to_student_folder(image_paths=image_paths, student_folder=student_folder))
        if capture_from_webcam:
            imported.extend(
                _capture_images_from_webcam(
                    student_folder=student_folder,
                    count=capture_count,
                    camera_index=camera_index,
                )
            )

        if not imported:
            raise RuntimeError("No images were provided or captured for the student")

        valid_count = 0
        for image_path in imported:
            valid, reason = _validate_single_face_image(image_path)
            if valid:
                valid_count += 1
                continue

            skipped.append(f"{image_path.name}: {reason}")
            image_path.unlink(missing_ok=True)

        if valid_count == 0:
            raise RuntimeError("All images were rejected. Add clear images with exactly one face.")

        report = rebuild_and_save_encodings(dataset_dir=dataset_path, cache_path=cache_path)
        student = StudentInfo(
            folder_name=folder_name,
            student_id=sid,
            student_name=name,
            image_count=valid_count,
        )
        return AddStudentResult(
            True,
            f"Student added successfully: {name} ({valid_count} valid images)",
            student,
            report,
            skipped,
        )
    except Exception as exc:
        shutil.rmtree(student_folder, ignore_errors=True)
        return AddStudentResult(
            False,
            f"Failed to add student: {exc}",
            None,
            None,
            skipped,
        )


def remove_student(
    student_ref: str,
    dataset_dir: str | Path,
    cache_path: str | Path,
) -> RemoveStudentResult:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return RemoveStudentResult(False, f"Dataset directory not found: {dataset_path}", None, None)

    target_folder = _find_student_folder(dataset_path=dataset_path, student_ref=student_ref)
    if target_folder is None:
        return RemoveStudentResult(
            False,
            f"Student not found or ambiguous reference: {student_ref}",
            None,
            None,
        )

    student_id, student_name = _parse_student_folder(target_folder.name)
    image_count = sum(
        1
        for image in target_folder.glob("*")
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    )
    removed_student = StudentInfo(
        folder_name=target_folder.name,
        student_id=student_id,
        student_name=student_name,
        image_count=image_count,
    )

    try:
        shutil.rmtree(target_folder)
        report = rebuild_and_save_encodings(dataset_dir=dataset_path, cache_path=cache_path)
        return RemoveStudentResult(
            True,
            f"Removed student: {removed_student.student_name}",
            removed_student,
            report,
        )
    except Exception as exc:
        return RemoveStudentResult(
            False,
            f"Failed to remove student: {exc}",
            None,
            None,
        )


def update_student_images(
    student_ref: str,
    dataset_dir: str | Path,
    cache_path: str | Path,
    new_image_paths: list[str] | None = None,
    capture_from_webcam: bool = False,
    capture_count: int = 3,
    camera_index: int = 0,
) -> AddStudentResult:
    dataset_path = Path(dataset_dir)
    target_folder = _find_student_folder(dataset_path=dataset_path, student_ref=student_ref)
    if target_folder is None:
        return AddStudentResult(False, f"Student not found: {student_ref}", None, None, [])

    student_id, student_name = _parse_student_folder(target_folder.name)

    skipped: list[str] = []
    imported: list[Path] = []
    if new_image_paths:
        imported.extend(_copy_images_to_student_folder(new_image_paths, target_folder))
    if capture_from_webcam:
        imported.extend(_capture_images_from_webcam(target_folder, capture_count, camera_index=camera_index))

    if not imported:
        return AddStudentResult(False, "No new images provided", None, None, [])

    valid_count = 0
    for image_path in imported:
        valid, reason = _validate_single_face_image(image_path)
        if valid:
            valid_count += 1
            continue

        skipped.append(f"{image_path.name}: {reason}")
        image_path.unlink(missing_ok=True)

    if valid_count == 0:
        return AddStudentResult(False, "No valid images were added", None, None, skipped)

    report = rebuild_and_save_encodings(dataset_dir=dataset_path, cache_path=cache_path)
    total_images = sum(
        1
        for image in target_folder.glob("*")
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    )
    student = StudentInfo(
        folder_name=target_folder.name,
        student_id=student_id,
        student_name=student_name,
        image_count=total_images,
    )
    return AddStudentResult(
        True,
        f"Updated images for {student_name}. Added {valid_count} valid images.",
        student,
        report,
        skipped,
    )

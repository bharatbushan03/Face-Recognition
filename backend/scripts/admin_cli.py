from __future__ import annotations

import argparse
from pathlib import Path

from backend.services.admin_service import (
    add_student,
    list_students,
    load_encodings,
    rebuild_and_save_encodings,
    remove_student,
    update_student_images,
)


def _print_students(dataset_dir: str) -> None:
    students = list_students(dataset_dir)
    if not students:
        print("No students found.")
        return

    print("\nStudents")
    print("-" * 72)
    print(f"{'Folder':28} {'ID':12} {'Name':20} {'Images':6}")
    print("-" * 72)
    for student in students:
        print(
            f"{student.folder_name[:28]:28} "
            f"{student.student_id[:12]:12} "
            f"{student.student_name[:20]:20} "
            f"{student.image_count:6}"
        )
    print("-" * 72)


def _prompt_paths(prompt: str) -> list[str]:
    raw = input(prompt).strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _show_encoding_cache_status(cache_file: str) -> None:
    try:
        store = load_encodings(cache_file)
        print(f"Encoding cache loaded successfully: {store.size} encodings")
    except Exception as exc:
        print(f"Encoding cache warning: {exc}")


def run_menu(dataset_dir: str, cache_file: str) -> None:
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)

    print("Face Attendance Admin CLI")
    print(f"Dataset directory: {Path(dataset_dir)}")
    print(f"Encoding cache: {Path(cache_file)}")
    _show_encoding_cache_status(cache_file)

    while True:
        print("\nChoose an option")
        print("1) View student list")
        print("2) Add student from image files")
        print("3) Add student from webcam capture")
        print("4) Update student images")
        print("5) Remove student")
        print("6) Rebuild/re-encode all faces")
        print("7) Exit")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            _print_students(dataset_dir)

        elif choice == "2":
            name = input("Student name: ").strip()
            sid = input("Student ID (optional): ").strip() or None
            image_paths = _prompt_paths("Image file paths (comma separated): ")

            result = add_student(
                student_name=name,
                student_id=sid,
                dataset_dir=dataset_dir,
                cache_path=cache_file,
                image_paths=image_paths,
                capture_from_webcam=False,
            )
            print(result.message)
            if result.skipped_images:
                print("Skipped images:")
                for item in result.skipped_images:
                    print(f"- {item}")

        elif choice == "3":
            name = input("Student name: ").strip()
            sid = input("Student ID (optional): ").strip() or None
            count_raw = input("How many images to capture (default 5): ").strip()
            count = int(count_raw) if count_raw else 5
            camera_raw = input("Camera index (default 0): ").strip()
            camera_index = int(camera_raw) if camera_raw else 0

            result = add_student(
                student_name=name,
                student_id=sid,
                dataset_dir=dataset_dir,
                cache_path=cache_file,
                capture_from_webcam=True,
                capture_count=count,
                camera_index=camera_index,
            )
            print(result.message)
            if result.skipped_images:
                print("Skipped images:")
                for item in result.skipped_images:
                    print(f"- {item}")

        elif choice == "4":
            student_ref = input("Student reference (ID, name, or folder): ").strip()
            print("Choose update source")
            print("1) Upload image files")
            print("2) Webcam capture")
            sub = input("Choice: ").strip()

            if sub == "1":
                image_paths = _prompt_paths("New image file paths (comma separated): ")
                result = update_student_images(
                    student_ref=student_ref,
                    dataset_dir=dataset_dir,
                    cache_path=cache_file,
                    new_image_paths=image_paths,
                )
            else:
                count_raw = input("How many images to capture (default 3): ").strip()
                count = int(count_raw) if count_raw else 3
                camera_raw = input("Camera index (default 0): ").strip()
                camera_index = int(camera_raw) if camera_raw else 0

                result = update_student_images(
                    student_ref=student_ref,
                    dataset_dir=dataset_dir,
                    cache_path=cache_file,
                    capture_from_webcam=True,
                    capture_count=count,
                    camera_index=camera_index,
                )

            print(result.message)
            if result.skipped_images:
                print("Skipped images:")
                for item in result.skipped_images:
                    print(f"- {item}")

        elif choice == "5":
            student_ref = input("Student reference to remove (ID, name, or folder): ").strip()
            confirm = input(f"Type YES to confirm removal of '{student_ref}': ").strip()
            if confirm != "YES":
                print("Removal cancelled.")
                continue

            result = remove_student(student_ref=student_ref, dataset_dir=dataset_dir, cache_path=cache_file)
            print(result.message)

        elif choice == "6":
            print("Rebuilding encodings...")
            report = rebuild_and_save_encodings(dataset_dir=dataset_dir, cache_path=cache_file)
            print(
                f"Done. Students: {report.students_found} | Images scanned: {report.images_scanned} | "
                f"Encodings: {report.encodings_created} | Skipped(no-face): {report.skipped_no_face} | "
                f"Skipped(errors): {report.skipped_errors}"
            )

        elif choice == "7":
            print("Goodbye.")
            break

        else:
            print("Invalid choice. Enter 1-7.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Admin CLI for face attendance student management")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/students",
        help="Directory containing student folders/images",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="models/known_face_encodings.npz",
        help="Encoding cache file path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_menu(dataset_dir=args.dataset_dir, cache_file=args.cache_file)

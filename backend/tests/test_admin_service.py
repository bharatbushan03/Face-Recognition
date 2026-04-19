from pathlib import Path

from backend.services.admin_service import (
    add_student,
    load_encodings,
    remove_student,
)
from backend.services.face_encoding_store import EncodingBuildReport


def _dummy_report(dataset_dir: Path) -> EncodingBuildReport:
    return EncodingBuildReport(
        dataset_dir=str(dataset_dir),
        students_found=1,
        images_scanned=1,
        encodings_created=1,
        skipped_no_face=0,
        skipped_errors=0,
    )


def test_add_student_rejects_duplicate_student_id(tmp_path):
    dataset_dir = tmp_path / "students"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "S001__Alice").mkdir()

    src_image = tmp_path / "alice.jpg"
    src_image.write_bytes(b"dummy")

    result = add_student(
        student_name="Bob",
        student_id="S001",
        dataset_dir=dataset_dir,
        cache_path=tmp_path / "encodings.npz",
        image_paths=[str(src_image)],
    )

    assert result.success is False
    assert "Duplicate student_id" in result.message


def test_add_student_with_no_valid_faces_rolls_back_folder(tmp_path, monkeypatch):
    from backend.services import admin_service

    dataset_dir = tmp_path / "students"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cache_file = tmp_path / "encodings.npz"

    src_image = tmp_path / "invalid.jpg"
    src_image.write_bytes(b"dummy")

    monkeypatch.setattr(admin_service, "_validate_single_face_image", lambda _path: (False, "No face detected"))

    result = add_student(
        student_name="Charlie",
        student_id="S010",
        dataset_dir=dataset_dir,
        cache_path=cache_file,
        image_paths=[str(src_image)],
    )

    assert result.success is False
    assert "All images were rejected" in result.message
    assert not (dataset_dir / "S010__Charlie").exists()


def test_add_then_remove_student_updates_dataset_consistently(tmp_path, monkeypatch):
    from backend.services import admin_service

    dataset_dir = tmp_path / "students"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cache_file = tmp_path / "encodings.npz"

    src_image = tmp_path / "face.jpg"
    src_image.write_bytes(b"dummy")

    monkeypatch.setattr(admin_service, "_validate_single_face_image", lambda _path: (True, None))
    monkeypatch.setattr(
        admin_service,
        "rebuild_and_save_encodings",
        lambda dataset_dir, cache_path, detection_model="hog", num_jitters=1: _dummy_report(Path(dataset_dir)),
    )

    added = add_student(
        student_name="Daisy",
        student_id="S222",
        dataset_dir=dataset_dir,
        cache_path=cache_file,
        image_paths=[str(src_image)],
    )
    assert added.success is True
    assert (dataset_dir / "S222__Daisy").exists()

    removed = remove_student(
        student_ref="S222",
        dataset_dir=dataset_dir,
        cache_path=cache_file,
    )
    assert removed.success is True
    assert not (dataset_dir / "S222__Daisy").exists()


def test_load_encodings_handles_corruption(tmp_path):
    cache_file = tmp_path / "encodings.npz"
    cache_file.write_text("not a valid npz", encoding="utf-8")

    try:
        load_encodings(cache_file)
        assert False, "Expected RuntimeError for corrupted cache"
    except RuntimeError as exc:
        assert "corrupted or unreadable" in str(exc)

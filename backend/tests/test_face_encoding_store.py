from pathlib import Path

from backend.services.face_encoding_store import _extract_student_name


def test_extract_student_name_with_id_prefix_folder(tmp_path):
    dataset_dir = tmp_path / "students"
    student_dir = dataset_dir / "S100__John_Doe"
    student_dir.mkdir(parents=True, exist_ok=True)

    image_path = student_dir / "img_001.jpg"
    image_path.write_bytes(b"dummy")

    name = _extract_student_name(image_path=image_path, dataset_dir=dataset_dir)
    assert name == "S100:John Doe"


def test_extract_student_name_without_id_prefix_folder(tmp_path):
    dataset_dir = tmp_path / "students"
    student_dir = dataset_dir / "Alice_Smith"
    student_dir.mkdir(parents=True, exist_ok=True)

    image_path = student_dir / "img_001.jpg"
    image_path.write_bytes(b"dummy")

    name = _extract_student_name(image_path=image_path, dataset_dir=dataset_dir)
    assert name == "Alice Smith"

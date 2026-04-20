from pathlib import Path
import numpy as np

from backend.services.face_encoding_store import _extract_student_name, KnownFaceStore


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


def test_match_batch_returns_expected_results():
    known_encodings = np.vstack(
        [
            np.zeros(128, dtype=np.float32),
            np.ones(128, dtype=np.float32) * 0.2,
        ]
    )
    store = KnownFaceStore(names=["Alice", "Bob"], encodings=known_encodings)

    queries = np.vstack(
        [
            np.zeros(128, dtype=np.float32),
            np.ones(128, dtype=np.float32) * 0.2,
            np.ones(128, dtype=np.float32) * 0.8,
        ]
    )

    results = store.match_batch(queries, tolerance=0.5, ambiguity_margin=0.01)

    assert len(results) == 3
    assert results[0].is_match is True and results[0].name == "Alice"
    assert results[1].is_match is True and results[1].name == "Bob"
    assert results[2].is_match is False


def test_match_batch_handles_empty_queries():
    store = KnownFaceStore.empty()
    assert store.match_batch([]) == []

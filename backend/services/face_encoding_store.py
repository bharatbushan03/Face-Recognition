from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import logging

import numpy as np


logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class EncodingBuildReport:
    dataset_dir: str
    students_found: int
    images_scanned: int
    encodings_created: int
    skipped_no_face: int
    skipped_errors: int


@dataclass(frozen=True)
class MatchResult:
    name: str | None
    distance: float
    confidence: float
    is_match: bool
    is_ambiguous: bool
    top_candidates: list[tuple[str, float]]


class KnownFaceStore:
    """Stores known student face encodings and supports nearest-distance matching."""

    def __init__(self, names: list[str], encodings: np.ndarray):
        self._names = list(names)
        self._encodings = np.asarray(encodings, dtype=np.float32)

        if len(self._names) == 0:
            self._encodings = np.empty((0, 128), dtype=np.float32)
            return

        if self._encodings.ndim != 2 or self._encodings.shape[1] != 128:
            raise ValueError("Known face encodings must be shaped as (N, 128)")
        if len(self._names) != self._encodings.shape[0]:
            raise ValueError("Names and encodings length mismatch")

    @property
    def names(self) -> list[str]:
        return list(self._names)

    @property
    def encodings(self) -> np.ndarray:
        return self._encodings.copy()

    @property
    def size(self) -> int:
        return len(self._names)

    @classmethod
    def empty(cls) -> "KnownFaceStore":
        return cls(names=[], encodings=np.empty((0, 128), dtype=np.float32))

    @classmethod
    def load_npz(cls, npz_path: str | Path) -> "KnownFaceStore":
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Encoding cache file not found: {path}")

        try:
            with np.load(path, allow_pickle=False) as data:
                if "encodings" not in data or "names" not in data:
                    raise ValueError("Missing required keys 'encodings' and/or 'names'")
                encodings = data["encodings"]
                names = data["names"].tolist()
        except Exception as exc:
            raise RuntimeError(f"Failed to load encoding cache from {path}: {exc}") from exc

        names = [str(name) for name in names]
        return cls(names=names, encodings=np.asarray(encodings, dtype=np.float32))

    def save_npz(self, npz_path: str | Path) -> None:
        path = Path(npz_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, encodings=self._encodings, names=np.array(self._names, dtype=str))

    @staticmethod
    def _validate_query_array(query_encodings: np.ndarray) -> np.ndarray:
        queries = np.asarray(query_encodings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2 or queries.shape[1] != 128:
            raise ValueError("Query encoding(s) must be shaped as (128,) or (N, 128)")
        return queries

    def _compute_distances(self, query_encodings: np.ndarray) -> np.ndarray:
        if self.size == 0:
            return np.empty((query_encodings.shape[0], 0), dtype=np.float32)

        # Vectorized L2 distance: shape (queries, known_faces).
        deltas = self._encodings[np.newaxis, :, :] - query_encodings[:, np.newaxis, :]
        return np.linalg.norm(deltas, axis=2)

    def _build_match_result(
        self,
        distances: np.ndarray,
        tolerance: float,
        ambiguity_margin: float,
        top_k: int,
    ) -> MatchResult:
        if distances.size == 0:
            return MatchResult(
                name=None,
                distance=1.0,
                confidence=0.0,
                is_match=False,
                is_ambiguous=False,
                top_candidates=[],
            )

        sorted_indices = np.argsort(distances)

        best_index = int(sorted_indices[0])
        best_distance = float(distances[best_index])
        best_name = self._names[best_index]
        confidence = max(0.0, 1.0 - min(best_distance, 1.0))

        top_candidates: list[tuple[str, float]] = []
        for index in sorted_indices[: max(1, top_k)]:
            top_candidates.append((self._names[int(index)], float(distances[int(index)])))

        is_ambiguous = False
        if len(sorted_indices) > 1:
            second_distance = float(distances[int(sorted_indices[1])])
            is_ambiguous = abs(second_distance - best_distance) < ambiguity_margin

        is_match = best_distance <= tolerance and not is_ambiguous

        return MatchResult(
            name=best_name if is_match else None,
            distance=best_distance,
            confidence=confidence,
            is_match=is_match,
            is_ambiguous=is_ambiguous,
            top_candidates=top_candidates,
        )

    def match(
        self,
        query_encoding: np.ndarray,
        tolerance: float = 0.48,
        ambiguity_margin: float = 0.03,
        top_k: int = 2,
    ) -> MatchResult:
        """
        Match an unknown face encoding to the nearest known encoding.

        Ambiguous matches happen when top-2 distances are too close.
        """
        queries = self._validate_query_array(np.asarray(query_encoding, dtype=np.float32))
        distances = self._compute_distances(queries)[0]
        return self._build_match_result(
            distances=distances,
            tolerance=tolerance,
            ambiguity_margin=ambiguity_margin,
            top_k=top_k,
        )

    def match_batch(
        self,
        query_encodings: list[np.ndarray] | np.ndarray,
        tolerance: float = 0.48,
        ambiguity_margin: float = 0.03,
        top_k: int = 2,
    ) -> list[MatchResult]:
        """Batch match multiple query encodings for lower per-frame recognition latency."""
        raw_queries = np.asarray(query_encodings, dtype=np.float32)
        if raw_queries.size == 0:
            return []

        queries = self._validate_query_array(raw_queries)
        distances_matrix = self._compute_distances(queries)
        return [
            self._build_match_result(
                distances=distances_matrix[row_index],
                tolerance=tolerance,
                ambiguity_margin=ambiguity_margin,
                top_k=top_k,
            )
            for row_index in range(distances_matrix.shape[0])
        ]


def _iter_images(dataset_dir: Path) -> Iterable[Path]:
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _largest_face_location(locations: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    # face_recognition locations are (top, right, bottom, left)
    return max(locations, key=lambda loc: max((loc[1] - loc[3]) * (loc[2] - loc[0]), 0))


def _extract_student_name(image_path: Path, dataset_dir: Path) -> str:
    relative = image_path.relative_to(dataset_dir)

    # Preferred: dataset/student_name/image1.jpg
    if len(relative.parts) > 1:
        folder = relative.parts[0]
        if "__" in folder:
            student_id, raw_name = folder.split("__", 1)
            clean_name = raw_name.replace("_", " ").strip() or raw_name
            return f"{student_id}:{clean_name}"
        return folder.replace("_", " ").strip() or folder

    # Fallback: studentname__image1.jpg or studentname_01.jpg
    stem = image_path.stem
    if "__" in stem:
        student_id, raw_name = stem.split("__", 1)
        clean_name = raw_name.replace("_", " ").strip() or raw_name
        return f"{student_id}:{clean_name}"
    if "_" in stem:
        return stem.split("_", 1)[0].replace("_", " ").strip()
    return stem.replace("_", " ").strip()


def build_known_face_store(
    dataset_dir: str | Path,
    detection_model: str = "hog",
    num_jitters: int = 1,
) -> tuple[KnownFaceStore, EncodingBuildReport]:
    """
    Build known face encodings from a student image dataset.

    Dataset structure supports:
      1) data/students/<student_name>/<images...>
      2) data/students/<student_name>__<anything>.jpg
    """
    try:
        import face_recognition
    except ImportError as exc:
        raise ImportError(
            "face_recognition is not installed. Run: python -m pip install face_recognition"
        ) from exc

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    names: list[str] = []
    encodings: list[np.ndarray] = []

    images_scanned = 0
    skipped_no_face = 0
    skipped_errors = 0
    student_names_seen: set[str] = set()

    for image_path in _iter_images(dataset_path):
        images_scanned += 1
        student_name = _extract_student_name(image_path, dataset_path)
        student_names_seen.add(student_name)

        try:
            image = face_recognition.load_image_file(str(image_path))
            locations = face_recognition.face_locations(image, model=detection_model)

            if not locations:
                skipped_no_face += 1
                logger.warning("No face found in %s", image_path)
                continue

            chosen_location = _largest_face_location(locations)
            image_encodings = face_recognition.face_encodings(
                image,
                known_face_locations=[chosen_location],
                num_jitters=num_jitters,
            )
            if not image_encodings:
                skipped_no_face += 1
                logger.warning("No encodings extracted from %s", image_path)
                continue

            names.append(student_name)
            encodings.append(np.asarray(image_encodings[0], dtype=np.float32))
        except Exception as exc:
            skipped_errors += 1
            logger.exception("Failed processing %s: %s", image_path, exc)

    if encodings:
        store = KnownFaceStore(names=names, encodings=np.vstack(encodings))
    else:
        store = KnownFaceStore.empty()

    report = EncodingBuildReport(
        dataset_dir=str(dataset_path),
        students_found=len(student_names_seen),
        images_scanned=images_scanned,
        encodings_created=store.size,
        skipped_no_face=skipped_no_face,
        skipped_errors=skipped_errors,
    )
    return store, report

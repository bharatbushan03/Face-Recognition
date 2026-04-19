from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
import csv
import threading


ATTENDANCE_HEADERS = [
    "student_key",
    "student_id",
    "student_name",
    "date",
    "time",
    "status",
    "session_id",
    "confidence",
    "source",
    "recorded_at",
]


@dataclass(frozen=True)
class AttendanceConfig:
    """Configuration for attendance persistence and duplicate policy."""

    csv_path: str = "attendance_logs/attendance.csv"
    dedupe_scope: str = "day"  # day | session
    status_present: str = "Present"
    session_id: str | None = None


@dataclass(frozen=True)
class AttendanceRecord:
    student_key: str
    student_id: str
    student_name: str
    date: str
    time: str
    status: str
    session_id: str
    confidence: float | None
    source: str
    recorded_at: str


class AttendanceManager:
    """
    Manages attendance records and prevents duplicate marks.

    Duplicate behavior:
      - dedupe_scope='day': one mark per student per date.
      - dedupe_scope='session': one mark per student per session.
    """

    def __init__(self, config: AttendanceConfig | None = None):
        self.config = config or AttendanceConfig()
        if self.config.dedupe_scope not in {"day", "session"}:
            raise ValueError("dedupe_scope must be one of: day, session")

        self.csv_path = Path(self.config.csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.session_id = self.config.session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self._active_date: date = date.today()
        self._marked_keys: set[str] = set()
        self._lock = threading.Lock()

        self._ensure_file()
        self._reload_marked_keys_for_active_date()

    @staticmethod
    def _normalize_student_key(student_name: str, student_id: str | None = None) -> str:
        if student_id and str(student_id).strip():
            return str(student_id).strip()
        return student_name.strip().lower()

    def _dedupe_key(self, student_key: str, for_date: str) -> str:
        if self.config.dedupe_scope == "session":
            return f"{student_key}|{for_date}|{self.session_id}"
        return f"{student_key}|{for_date}"

    def _ensure_file(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return

        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ATTENDANCE_HEADERS)
            writer.writeheader()

    def _reload_marked_keys_for_active_date(self) -> None:
        self._marked_keys.clear()
        target_date = self._active_date.isoformat()

        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return

        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("date") != target_date:
                    continue

                student_key = row.get("student_key", "").strip()
                if not student_key:
                    continue

                if self.config.dedupe_scope == "session":
                    if row.get("session_id") == self.session_id:
                        self._marked_keys.add(self._dedupe_key(student_key, target_date))
                else:
                    self._marked_keys.add(self._dedupe_key(student_key, target_date))

    def _refresh_day_if_needed(self, now: datetime) -> None:
        current = now.date()
        if current == self._active_date:
            return

        self._active_date = current
        self._reload_marked_keys_for_active_date()

    def mark_present(
        self,
        student_name: str,
        student_id: str | None = None,
        confidence: float | None = None,
        source: str = "face_recognition",
    ) -> tuple[bool, AttendanceRecord | None]:
        """
        Mark a student present if not already marked in dedupe scope.

        Returns:
            (True, record)  when a new mark is written
            (False, None)   when duplicate or invalid input
        """
        clean_name = student_name.strip()
        if not clean_name:
            return False, None

        with self._lock:
            now = datetime.now()
            self._refresh_day_if_needed(now)

            day_str = now.date().isoformat()
            time_str = now.strftime("%H:%M:%S")
            recorded_at = now.isoformat(timespec="seconds")

            student_key = self._normalize_student_key(clean_name, student_id=student_id)
            dedupe_key = self._dedupe_key(student_key, day_str)

            if dedupe_key in self._marked_keys:
                return False, None

            record = AttendanceRecord(
                student_key=student_key,
                student_id=(student_id or "").strip(),
                student_name=clean_name,
                date=day_str,
                time=time_str,
                status=self.config.status_present,
                session_id=self.session_id,
                confidence=confidence,
                source=source,
                recorded_at=recorded_at,
            )

            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=ATTENDANCE_HEADERS)
                writer.writerow(
                    {
                        "student_key": record.student_key,
                        "student_id": record.student_id,
                        "student_name": record.student_name,
                        "date": record.date,
                        "time": record.time,
                        "status": record.status,
                        "session_id": record.session_id,
                        "confidence": "" if record.confidence is None else f"{record.confidence:.4f}",
                        "source": record.source,
                        "recorded_at": record.recorded_at,
                    }
                )

            self._marked_keys.add(dedupe_key)
            return True, record

    def is_marked_today(self, student_name: str, student_id: str | None = None) -> bool:
        now = datetime.now()
        self._refresh_day_if_needed(now)

        student_key = self._normalize_student_key(student_name, student_id=student_id)
        key = self._dedupe_key(student_key, now.date().isoformat())
        return key in self._marked_keys

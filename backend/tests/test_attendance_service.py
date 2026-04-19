from pathlib import Path
import csv

from backend.services.attendance_service import AttendanceConfig, AttendanceManager


def _read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_mark_once_per_day(tmp_path):
    csv_path = tmp_path / "attendance.csv"
    manager = AttendanceManager(
        AttendanceConfig(csv_path=str(csv_path), dedupe_scope="day", session_id="session_A")
    )

    marked_first, record_first = manager.mark_present("Alice", student_id="S001", confidence=0.91)
    marked_second, record_second = manager.mark_present("Alice", student_id="S001", confidence=0.87)

    assert marked_first is True
    assert record_first is not None
    assert marked_second is False
    assert record_second is None

    rows = _read_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["student_id"] == "S001"
    assert rows[0]["student_name"] == "Alice"
    assert rows[0]["status"] == "Present"


def test_restart_still_prevents_duplicate_in_day_scope(tmp_path):
    csv_path = tmp_path / "attendance.csv"

    manager_first = AttendanceManager(
        AttendanceConfig(csv_path=str(csv_path), dedupe_scope="day", session_id="session_1")
    )
    marked, _ = manager_first.mark_present("Bob", student_id="S002")
    assert marked is True

    # Simulate restart by creating a new manager instance for same file.
    manager_after_restart = AttendanceManager(
        AttendanceConfig(csv_path=str(csv_path), dedupe_scope="day", session_id="session_2")
    )
    marked_again, _ = manager_after_restart.mark_present("Bob", student_id="S002")
    assert marked_again is False

    rows = _read_rows(csv_path)
    assert len(rows) == 1


def test_session_scope_allows_new_session_mark(tmp_path):
    csv_path = tmp_path / "attendance.csv"

    manager_session_one = AttendanceManager(
        AttendanceConfig(csv_path=str(csv_path), dedupe_scope="session", session_id="morning")
    )
    first_marked, _ = manager_session_one.mark_present("Carol", student_id="S003")
    assert first_marked is True

    # Same day, different session should allow one more mark in session scope.
    manager_session_two = AttendanceManager(
        AttendanceConfig(csv_path=str(csv_path), dedupe_scope="session", session_id="afternoon")
    )
    second_marked, _ = manager_session_two.mark_present("Carol", student_id="S003")
    assert second_marked is True

    rows = _read_rows(csv_path)
    assert len(rows) == 2
    assert rows[0]["session_id"] == "morning"
    assert rows[1]["session_id"] == "afternoon"

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Allow running via "streamlit run streamlit_dashboard.py" from backend/ui.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from backend.ui.dashboard_runtime import AttendanceDashboardRuntime, DashboardRuntimeConfig
from backend.utils.logging_config import configure_logging


configure_logging()


st.set_page_config(page_title="Face Recognition Attendance Dashboard", layout="wide")


# Keep runtime objects in session state so camera and recognition state survive reruns.
if "runtime" not in st.session_state:
    st.session_state.runtime = None
if "running" not in st.session_state:
    st.session_state.running = False
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "runtime_tuning" not in st.session_state:
    st.session_state.runtime_tuning = {
        "recognition_tolerance": 0.48,
        "recognition_detect_every_n_frames": 2,
        "recognition_process_every_n_frames": 3,
        "recognition_max_faces_per_frame": 4,
        "recognition_enable_low_light_enhancement": True,
        "detection_process_width": 512,
        "camera_frame_width": 640,
        "camera_frame_height": 360,
        "ui_refresh_fps": 12,
        "display_smoothing_alpha": 0.22,
        "table_refresh_seconds": 1.0,
    }


def _build_runtime(force_refresh_cache: bool = False) -> AttendanceDashboardRuntime:
    tuning = st.session_state.runtime_tuning
    return AttendanceDashboardRuntime(
        DashboardRuntimeConfig(
            dataset_dir="data/students",
            cache_file="models/known_face_encodings.npz",
            refresh_cache=force_refresh_cache,
            attendance_file="attendance_logs/attendance.csv",
            dedupe_scope="day",
            camera_index=0,
            camera_buffer_size=1,
            frame_width=int(tuning["camera_frame_width"]),
            frame_height=int(tuning["camera_frame_height"]),
            mirror=True,
            display_unknown_warnings=False,
            display_smoothing_alpha=float(tuning["display_smoothing_alpha"]),
            detection_process_width=int(tuning["detection_process_width"]),
            recognition_tolerance=float(tuning["recognition_tolerance"]),
            recognition_detect_every_n_frames=int(tuning["recognition_detect_every_n_frames"]),
            recognition_process_every_n_frames=int(tuning["recognition_process_every_n_frames"]),
            recognition_max_faces_per_frame=int(tuning["recognition_max_faces_per_frame"]),
            recognition_enable_low_light_enhancement=bool(
                tuning["recognition_enable_low_light_enhancement"]
            ),
        )
    )


def _ensure_runtime() -> AttendanceDashboardRuntime:
    runtime: AttendanceDashboardRuntime | None = st.session_state.runtime
    if runtime is None:
        runtime = _build_runtime(force_refresh_cache=False)
        st.session_state.runtime = runtime
    return runtime


def _stop_camera_if_running() -> None:
    runtime: AttendanceDashboardRuntime | None = st.session_state.runtime
    if runtime is not None:
        runtime.stop_camera()


def _camera_refresh_interval_seconds() -> float | None:
    if not st.session_state.running:
        return None

    ui_refresh_fps = max(4, int(st.session_state.runtime_tuning.get("ui_refresh_fps", 12)))
    return 1.0 / float(ui_refresh_fps)


def _table_refresh_interval_seconds() -> float | None:
    if not st.session_state.running:
        return None

    return max(0.25, float(st.session_state.runtime_tuning.get("table_refresh_seconds", 1.0)))


st.title("Face Recognition Attendance Dashboard")
st.caption("Live camera monitoring, real-time attendance updates, and simple operator controls.")

try:
    runtime = _ensure_runtime()
except Exception as exc:
    st.error(f"Failed to initialize dashboard runtime: {exc}")
    st.stop()

camera_col, control_col = st.columns([2.2, 1.0])

with control_col:
    st.subheader("Control Panel")

    with st.expander("Performance & Accuracy Tuning", expanded=False):
        tune_tolerance = st.slider(
            "Recognition threshold (lower is stricter)",
            min_value=0.35,
            max_value=0.70,
            value=float(st.session_state.runtime_tuning["recognition_tolerance"]),
            step=0.01,
        )
        tune_detect_every_n = st.slider(
            "Detect every N frames",
            min_value=1,
            max_value=4,
            value=int(st.session_state.runtime_tuning["recognition_detect_every_n_frames"]),
            step=1,
            help="Higher values increase FPS by reusing the previous detection briefly between updates.",
        )
        tune_process_every_n = st.slider(
            "Encode every N frames",
            min_value=1,
            max_value=6,
            value=int(st.session_state.runtime_tuning["recognition_process_every_n_frames"]),
            step=1,
        )
        tune_max_faces = st.slider(
            "Max faces encoded per frame",
            min_value=1,
            max_value=12,
            value=int(st.session_state.runtime_tuning["recognition_max_faces_per_frame"]),
            step=1,
        )
        tune_detection_width = st.slider(
            "Detection process width",
            min_value=320,
            max_value=1280,
            value=int(st.session_state.runtime_tuning["detection_process_width"]),
            step=32,
        )
        tune_frame_width = st.slider(
            "Camera frame width",
            min_value=320,
            max_value=1280,
            value=int(st.session_state.runtime_tuning["camera_frame_width"]),
            step=32,
        )
        tune_frame_height = st.slider(
            "Camera frame height",
            min_value=240,
            max_value=720,
            value=int(st.session_state.runtime_tuning["camera_frame_height"]),
            step=16,
        )
        tune_ui_refresh_fps = st.slider(
            "UI refresh FPS",
            min_value=4,
            max_value=24,
            value=int(st.session_state.runtime_tuning["ui_refresh_fps"]),
            step=1,
            help="Lower values reduce CPU load and usually improve camera smoothness.",
        )
        tune_display_smoothing = st.slider(
            "Frame smoothing (anti-flicker)",
            min_value=0.0,
            max_value=0.6,
            value=float(st.session_state.runtime_tuning["display_smoothing_alpha"]),
            step=0.01,
            help="Higher values keep more of the previous frame, which reduces flicker but can add ghosting.",
        )
        tune_low_light = st.checkbox(
            "Enable low-light enhancement",
            value=bool(st.session_state.runtime_tuning["recognition_enable_low_light_enhancement"]),
        )
        apply_tuning = st.button("Apply Tuning")

        if apply_tuning:
            st.session_state.runtime_tuning = {
                "recognition_tolerance": tune_tolerance,
                "recognition_detect_every_n_frames": tune_detect_every_n,
                "recognition_process_every_n_frames": tune_process_every_n,
                "recognition_max_faces_per_frame": tune_max_faces,
                "recognition_enable_low_light_enhancement": tune_low_light,
                "detection_process_width": tune_detection_width,
                "camera_frame_width": tune_frame_width,
                "camera_frame_height": tune_frame_height,
                "ui_refresh_fps": tune_ui_refresh_fps,
                "display_smoothing_alpha": tune_display_smoothing,
                "table_refresh_seconds": float(st.session_state.runtime_tuning["table_refresh_seconds"]),
            }
            _stop_camera_if_running()
            st.session_state.running = False
            st.session_state.runtime = None
            runtime = _ensure_runtime()
            st.success("Tuning applied. Camera can be started with the updated settings.")

    start_clicked = st.button("Start Camera", type="primary", disabled=st.session_state.running)
    stop_clicked = st.button("Stop Camera", disabled=not st.session_state.running)
    rebuild_clicked = st.button("Rebuild Face Cache")

    if start_clicked:
        try:
            runtime.sync_known_faces(force_refresh_cache=False)
            runtime.start_camera()
            st.session_state.running = True
            st.session_state.last_error = ""
            st.rerun()
        except Exception as exc:
            st.session_state.running = False
            st.session_state.last_error = str(exc)

    if stop_clicked:
        _stop_camera_if_running()
        st.session_state.running = False
        st.rerun()

    if rebuild_clicked:
        # Rebuilding can take time; stop camera first for a cleaner reset.
        _stop_camera_if_running()
        st.session_state.running = False
        try:
            loaded = runtime.sync_known_faces(force_refresh_cache=True)
            st.success(f"Face cache refreshed. Loaded encodings: {loaded}")
        except Exception as exc:
            st.session_state.last_error = f"Failed to rebuild face cache: {exc}"

    @st.fragment(run_every=1.0 if st.session_state.running else None)
    def _render_system_stats() -> None:
        current_runtime = _ensure_runtime()

        if st.session_state.last_error:
            st.error(st.session_state.last_error)

        st.markdown("### System Stats")
        st.metric("Total Present Today", current_runtime.total_present_today)
        st.write(f"Detection Backend: {current_runtime.detection_backend}")
        st.write(f"Known Faces Loaded: {current_runtime.known_faces_count}")
        st.write(f"Session Time: {datetime.now().strftime('%H:%M:%S')}")

        if current_runtime.known_faces_count == 0:
            st.warning(
                "No registered users found yet. Register users from FastAPI (/api/face/register), "
                "then click 'Rebuild Face Cache'."
            )

        csv_bytes = current_runtime.get_attendance_csv_bytes()
        st.download_button(
            label="Download Attendance CSV",
            data=csv_bytes if csv_bytes else b"",
            file_name="attendance.csv",
            mime="text/csv",
            disabled=csv_bytes is None,
            help="Download is enabled after at least one attendance record exists.",
        )

    _render_system_stats()

with camera_col:
    @st.fragment(run_every=_camera_refresh_interval_seconds())
    def _render_camera_feed() -> None:
        st.subheader("Camera Feed Panel")

        current_runtime = _ensure_runtime()
        if not st.session_state.running:
            st.info("Camera is stopped. Click Start Camera to begin live monitoring.")
            return

        try:
            frame_update = current_runtime.process_next_frame()
        except Exception as exc:
            _stop_camera_if_running()
            st.session_state.running = False
            st.session_state.last_error = str(exc)
            st.rerun()
            return

        st.image(frame_update.frame_rgb, channels="RGB", use_container_width=True)

        if frame_update.notice:
            st.warning(frame_update.notice)
        else:
            st.caption(
                f"Faces detected: {frame_update.total_faces} | "
                f"Known: {frame_update.known_faces} | Unknown: {frame_update.unknown_faces}"
            )

        for event in frame_update.new_attendance_events:
            st.toast(f"Marked present: {event['student_name']} at {event['time']}")

    _render_camera_feed()

attendance_section = st.container(border=True)
with attendance_section:
    @st.fragment(run_every=_table_refresh_interval_seconds())
    def _render_attendance_table() -> None:
        st.subheader("Real-Time Attendance Table")
        search_query = st.text_input(
            "Search by student name or ID",
            value="",
            key="attendance_table_search_query",
        )

        rows = _ensure_runtime().get_attendance_rows(search_query=search_query)
        if rows:
            st.dataframe(rows, width="stretch", height=300)
        else:
            st.info("No attendance has been marked yet.")

    _render_attendance_table()

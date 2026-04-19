from __future__ import annotations

from datetime import datetime
import time

import streamlit as st

from backend.ui.dashboard_runtime import AttendanceDashboardRuntime, DashboardRuntimeConfig


st.set_page_config(page_title="Face Recognition Attendance Dashboard", layout="wide")


# Keep runtime objects in session state so camera and recognition state survive reruns.
if "runtime" not in st.session_state:
    st.session_state.runtime = None
if "running" not in st.session_state:
    st.session_state.running = False
if "last_error" not in st.session_state:
    st.session_state.last_error = ""


def _build_runtime(force_refresh_cache: bool = False) -> AttendanceDashboardRuntime:
    return AttendanceDashboardRuntime(
        DashboardRuntimeConfig(
            dataset_dir="data/students",
            cache_file="models/known_face_encodings.npz",
            refresh_cache=force_refresh_cache,
            attendance_file="attendance_logs/attendance.csv",
            dedupe_scope="day",
            camera_index=0,
            frame_width=960,
            frame_height=540,
            mirror=True,
            display_unknown_warnings=False,
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

    start_clicked = st.button("Start Camera", type="primary", disabled=st.session_state.running)
    stop_clicked = st.button("Stop Camera", disabled=not st.session_state.running)
    rebuild_clicked = st.button("Rebuild Face Cache")

    if start_clicked:
        try:
            runtime.start_camera()
            st.session_state.running = True
            st.session_state.last_error = ""
        except Exception as exc:
            st.session_state.running = False
            st.session_state.last_error = str(exc)

    if stop_clicked:
        _stop_camera_if_running()
        st.session_state.running = False

    if rebuild_clicked:
        # Rebuilding can take time; stop camera first for a cleaner reset.
        _stop_camera_if_running()
        st.session_state.running = False
        st.session_state.runtime = None
        try:
            st.session_state.runtime = _build_runtime(force_refresh_cache=True)
            runtime = st.session_state.runtime
            st.success("Face encoding cache rebuilt successfully.")
        except Exception as exc:
            st.session_state.last_error = f"Failed to rebuild face cache: {exc}"

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    st.markdown("### System Stats")
    st.metric("Total Present Today", runtime.total_present_today)
    st.write(f"Detection Backend: {runtime.detection_backend}")
    st.write(f"Session Time: {datetime.now().strftime('%H:%M:%S')}")

    csv_bytes = runtime.get_attendance_csv_bytes()
    st.download_button(
        label="Download Attendance CSV",
        data=csv_bytes if csv_bytes else b"",
        file_name="attendance.csv",
        mime="text/csv",
        disabled=csv_bytes is None,
        help="Download is enabled after at least one attendance record exists.",
    )

with camera_col:
    st.subheader("Camera Feed Panel")
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    if not st.session_state.running:
        frame_placeholder.info("Camera is stopped. Click Start Camera to begin live monitoring.")

attendance_section = st.container(border=True)
with attendance_section:
    st.subheader("Real-Time Attendance Table")
    search_query = st.text_input("Search by student name or ID", value="")

    rows = runtime.get_attendance_rows(search_query=search_query)
    if rows:
        st.dataframe(rows, use_container_width=True, height=300)
    else:
        st.info("No attendance has been marked yet.")

if st.session_state.running:
    try:
        frame_update = runtime.process_next_frame()

        frame_placeholder.image(frame_update.frame_rgb, channels="RGB", use_container_width=True)

        if frame_update.notice:
            status_placeholder.warning(frame_update.notice)
        else:
            status_placeholder.success(
                f"Faces detected: {frame_update.total_faces} | "
                f"Known: {frame_update.known_faces} | Unknown: {frame_update.unknown_faces}"
            )

        for event in frame_update.new_attendance_events:
            st.success(
                f"Marked present: {event['student_name']} at {event['time']} on {event['date']}"
            )

        # Moderate refresh interval keeps UI responsive without overloading CPU.
        time.sleep(0.03)
        st.rerun()
    except Exception as exc:
        _stop_camera_if_running()
        st.session_state.running = False
        st.session_state.last_error = str(exc)
        st.rerun()

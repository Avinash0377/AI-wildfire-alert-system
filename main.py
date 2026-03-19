import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import threading
import time
import streamlit as st
import tempfile
from datetime import datetime

from config import (
    MODEL_PATH, INFERENCE_IMG_SIZE, IOU_THRESHOLD,
    FRAME_WIDTH, FRAME_HEIGHT, FRAME_SKIP,
    DEFAULT_RECEIVER_EMAIL, DEFAULT_EMAIL_COOLDOWN,
    DEFAULT_CONFIDENCE_THRESHOLD, CONSECUTIVE_FRAMES_REQUIRED,
    SESSION_DEFAULTS,
)
from utils import send_email, get_location, enhance_frame, cleanup_temp_file

# ======================== PAGE CONFIG ========================
st.set_page_config(page_title="Fire Detection System", layout="wide")

# ======================== SESSION STATE ========================
for key, val in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

if st.session_state.live_location is None:
    st.session_state.live_location = get_location()

# ======================== CUSTOM CSS ========================
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    .main-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        color: #FF5733;
        margin-bottom: 0.5rem;
        padding: 0.5rem 0;
    }
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #e0e0e0;
        font-size: 1.1rem;
        border-bottom: 1px solid #333;
        padding-bottom: 0.3rem;
    }
    .stButton > button {
        font-weight: 600;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    div[data-testid="column"]:first-child .stButton > button {
        background-color: #2e7d32;
        color: white;
        border: none;
    }
    div[data-testid="column"]:first-child .stButton > button:hover {
        background-color: #1b5e20;
    }
    div[data-testid="column"]:last-child .stButton > button {
        background-color: #c62828;
        color: white;
        border: none;
    }
    div[data-testid="column"]:last-child .stButton > button:hover {
        background-color: #b71c1c;
    }
    div[data-testid="stMetric"] {
        background: #1e1e2f;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
    }
    div[data-testid="stMetric"] label {
        color: #aaa;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #fff;
        font-size: 1.4rem;
    }
    .stFileUploader > div {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ======================== SIDEBAR (Settings only) ========================
with st.sidebar:
    st.header("Settings")

    st.subheader("Email Alert")
    receiver_email = st.text_input("Receiver Email", value=DEFAULT_RECEIVER_EMAIL)
    email_cooldown = st.number_input(
        "Email Cooldown (seconds)",
        min_value=10, max_value=600,
        value=DEFAULT_EMAIL_COOLDOWN, step=10
    )

    st.subheader("Detection")
    confidence_threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=0, max_value=100,
        value=DEFAULT_CONFIDENCE_THRESHOLD, step=5
    )

    st.divider()
    if st.button("Reset Statistics", use_container_width=True):
        st.session_state.fire_count = 0
        st.session_state.detection_log = []
        st.session_state.fire_consecutive_frames = 0
        st.rerun()

    st.divider()
    st.caption("AI Wildfire Alert System v1.0")

# ======================== MAIN AREA ========================
st.markdown('<div class="main-title">Fire Detection Dashboard</div>', unsafe_allow_html=True)

# ======================== SOURCE SELECTION ========================
st.subheader("Select Source")
source_col1, source_col2 = st.columns([1, 2])

with source_col1:
    source_option = st.radio(
        "Input Type",
        ["Webcam", "RTSP / IP Camera", "Upload Video"],
        index=2,
        label_visibility="collapsed"
    )

with source_col2:
    uploaded_file = None
    rtsp_url = ""
    if source_option == "Upload Video":
        uploaded_file = st.file_uploader(
            "Upload Video File",
            type=["mp4", "avi", "mkv", "mpeg4"],
            key="main_file_uploader"
        )
    elif source_option == "RTSP / IP Camera":
        rtsp_url = st.text_input(
            "Camera URL",
            placeholder="rtsp://username:password@ip:port/stream"
        )
    elif source_option == "Webcam":
        st.info("Webcam will be used as the video source.")

# ======================== START / STOP BUTTONS ========================
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    start_clicked = st.button("Start Detection", use_container_width=True)
with btn_col2:
    stop_clicked = st.button("Stop Detection", use_container_width=True)

# Handle start
if start_clicked:
    if source_option == "Webcam":
        st.session_state.video_source = 0
        st.session_state.video_source_type = "webcam"
        st.session_state.detection_running = True
    elif source_option == "RTSP / IP Camera":
        if rtsp_url.strip():
            st.session_state.video_source = rtsp_url.strip()
            st.session_state.video_source_type = "rtsp"
            st.session_state.detection_running = True
        else:
            st.warning("Please enter an RTSP / IP Camera URL above.")
    elif source_option == "Upload Video":
        if uploaded_file is not None:
            cleanup_temp_file()
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()
            st.session_state.video_source = tfile.name
            st.session_state.temp_file_path = tfile.name
            st.session_state.video_source_type = "file"
            st.session_state.detection_running = True
        else:
            st.warning("Please upload a video file first.")

# Handle stop
if stop_clicked:
    st.session_state.detection_running = False

# ======================== LOAD YOLO MODEL ========================
try:
    model = YOLO(MODEL_PATH)
    names = model.model.names
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ======================== DETECTION LOOP ========================
stframe = st.empty()
status_placeholder = st.empty()

if st.session_state.detection_running and st.session_state.video_source is not None:
    if st.session_state.video_source == 0:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(st.session_state.video_source)

    if not cap.isOpened():
        st.error("Could not open video source. Please check the URL or file.")
        st.session_state.detection_running = False
    else:
        source_label = st.session_state.video_source_type or "unknown"
        status_placeholder.info(f"Detection running  |  Source: {source_label}")
        count = 0
        conf_threshold = confidence_threshold / 100.0

        while st.session_state.detection_running:
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            if count % FRAME_SKIP != 0:
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Preprocess frame for better fire visibility
            processed_frame = enhance_frame(frame)

            # Overlay timestamp & location for live sources
            if st.session_state.video_source_type in ("webcam", "rtsp"):
                if count % 60 == 0:
                    st.session_state.live_location = get_location()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                location = st.session_state.live_location
                cv2.putText(frame, f"Time: {timestamp}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Location: {location}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Run YOLO on preprocessed frame
            results = model.track(
                processed_frame, persist=True,
                imgsz=INFERENCE_IMG_SIZE,
                conf=conf_threshold,
                iou=IOU_THRESHOLD
            )

            try:
                fire_detected_this_frame = False

                # Draw segmentation masks if available
                if results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()
                    for mask in masks:
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        colored_mask = np.zeros_like(frame)
                        colored_mask[:, :, 2] = (mask_resized * 255).astype(np.uint8)
                        colored_mask[:, :, 1] = (mask_resized * 50).astype(np.uint8)
                        frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.4, 0)

                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.int().cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()
                    confs = results[0].boxes.conf.cpu().tolist()
                    track_ids = (
                        results[0].boxes.id.int().cpu().tolist()
                        if results[0].boxes.id is not None
                        else [-1] * len(boxes)
                    )

                    for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
                        c = names[class_id]

                        if st.session_state.video_source_type in ("webcam", "rtsp") and 'fire' not in c.lower():
                            continue

                        x1, y1, x2, y2 = box
                        conf_pct = conf * 100

                        # Color-coded bounding boxes
                        if 'fire' in c.lower():
                            box_color = (0, 0, 255)
                            thickness = 3
                        else:
                            box_color = (0, 255, 0)
                            thickness = 2

                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
                        cvzone.putTextRect(frame, f'ID:{track_id}', (x2, y2), 1, 1)
                        cvzone.putTextRect(frame, f'{c} {conf_pct:.0f}%', (x1, y1), 1, 1)

                        if 'fire' in c.lower():
                            fire_detected_this_frame = True
                            st.session_state.fire_consecutive_frames += 1

                            # Temporal confirmation
                            if st.session_state.fire_consecutive_frames >= CONSECUTIVE_FRAMES_REQUIRED:
                                st.session_state.fire_count += 1
                                det_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                log_entry = {
                                    "class": c,
                                    "confidence": f"{conf_pct:.1f}%",
                                    "box": str(box),
                                    "timestamp": det_timestamp,
                                    "location": st.session_state.live_location
                                }
                                st.session_state.detection_log.append(log_entry)

                                current_time = time.time()
                                if current_time - st.session_state.last_email_time > email_cooldown:
                                    thread = threading.Thread(
                                        target=send_email,
                                        args=(receiver_email, frame.copy()),
                                        kwargs={
                                            "location": st.session_state.live_location,
                                            "timestamp": det_timestamp,
                                            "confidence": conf_pct
                                        }
                                    )
                                    thread.start()
                                    st.session_state.email_threads.append(thread)
                                    st.session_state.last_email_time = current_time

                # Reset consecutive counter if no fire in this frame
                if not fire_detected_this_frame:
                    st.session_state.fire_consecutive_frames = 0

            except Exception as e:
                st.error(f"Detection error: {e}")

            stframe.image(frame, channels="BGR")
            time.sleep(0.03)

        cap.release()
        cleanup_temp_file()
        st.session_state.detection_running = False
        status_placeholder.success("Detection stopped.")

# ======================== STATISTICS ========================
st.divider()
col_stat1, col_stat2 = st.columns(2)
with col_stat1:
    st.metric("Total Fires Detected", st.session_state.fire_count)
with col_stat2:
    st.metric("Current Location", st.session_state.live_location or "Unknown")

if st.session_state.detection_log:
    st.subheader("Detection Log")
    st.dataframe(st.session_state.detection_log, use_container_width=True)

# ======================== CLEANUP THREADS ========================
for thread in st.session_state.email_threads:
    if thread.is_alive():
        thread.join(timeout=1)
# "python -m streamlit run main.py" to run
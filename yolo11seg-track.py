import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import threading
import time
import streamlit as st
import tempfile
import requests
from datetime import datetime

# ---------------- Email Sending Function ----------------
def send_email(receiver_email, frame, max_retries=3, delay=5):
    attempt = 0
    while attempt < max_retries:
        try:
            # Set up SMTP server and login
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login('sudhimallaavinash00@gmail.com', 'qlpu gzfj ldlu abos')

            # Create email message with image attachment
            msg = MIMEMultipart()
            msg['From'] = 'sudhimallaavinash00@gmail.com'
            msg['To'] = receiver_email
            msg['Subject'] = 'Fire Detected'
            _, buffer = cv2.imencode('.jpg', frame)
            img_data = buffer.tobytes()
            img = MIMEImage(img_data, name="Fire.jpg")
            img.add_header('Content-Disposition', 'attachment', filename="noname.jpg")
            msg.attach(img)

            # Send the email
            server.send_message(msg)
            st.write("Email sent successfully.")
            break  # Email sent successfully, exit loop

        except Exception as e:
            attempt += 1
            st.write(f"Failed to send email: {e}. Retrying {attempt}/{max_retries}...")
            time.sleep(delay)

        finally:
            try:
                server.quit()
            except Exception:
                pass

    if attempt == max_retries:
        st.write("Max retries reached. Could not send email.")

# ---------------- Location Function ----------------
def get_location():
    try:
        # Using ip-api.com for location without API token
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        city = data.get('city', 'Unknown')
        region = data.get('regionName', 'Unknown')
        country = data.get('country', 'Unknown')

        # Adjust region if city is Hyderabad but region returned is Andhra Pradesh
        if city.lower() == "hyderabad" and region.lower() == "andhra pradesh":
            region = "Telangana"

        location = f"{city}, {region}, {country}"
        return location
    except Exception as e:
        st.write("Error fetching location:", e)
        return "Location Unavailable"

# ---------------- Load YOLO Model ----------------
model = YOLO("best.pt")
names = model.model.names

# ---------------- Initialize Session State ----------------
if 'detection_running' not in st.session_state:
    st.session_state.detection_running = False
if 'video_source' not in st.session_state:
    st.session_state.video_source = None
if 'email_threads' not in st.session_state:
    st.session_state.email_threads = []
if 'live_location' not in st.session_state:
    st.session_state.live_location = get_location()

st.title("Fire Detection System")

# ---------------- Streamlit UI Controls ----------------
col1, col2, col3 = st.columns(3)
if col1.button("Live Detection"):
    st.session_state.video_source = 0  # Use webcam
    st.session_state.detection_running = True

uploaded_file = col2.file_uploader("Open File", type=["mp4", "avi", "mkv"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.session_state.video_source = tfile.name
    st.session_state.detection_running = True

if col3.button("Stop Detection"):
    st.session_state.detection_running = False

# ---------------- Detection Loop ----------------
stframe = st.empty()  # Placeholder for video stream display

if st.session_state.detection_running and st.session_state.video_source is not None:
    cap = cv2.VideoCapture(st.session_state.video_source)
    count = 0
    while st.session_state.detection_running:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        # Process every second frame to reduce load
        if count % 2 != 0:
            continue

        frame = cv2.resize(frame, (1020, 500))

        # For live detection, overlay timestamp and update location periodically
        if st.session_state.video_source == 0:
            if count % 60 == 0:
                st.session_state.live_location = get_location()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            location = st.session_state.live_location
            cv2.putText(frame, f"Time: {timestamp}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Location: {location}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Run YOLO tracking on the frame
        results = model.track(frame, persist=True)

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = [-1] * len(boxes)
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                c = names[class_id]
                # For live detection, process only fire detections
                if st.session_state.video_source == 0 and 'fire' not in c.lower():
                    continue

                x1, y1, x2, y2 = box
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{track_id}', (x2, y2), 1, 1)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

                # Check if the detected class includes 'fire'
                if 'fire' in c.lower():
                    st.write(f"Detected {c} at bounding box {box}")
                    receiver_email = "sudhimallaavinash03@gmail.com"
                    # Start email sending in a new thread
                    email_thread = threading.Thread(target=send_email, args=(receiver_email, frame.copy()))
                    st.session_state.email_threads.append(email_thread)
                    email_thread.start()

        stframe.image(frame, channels="BGR")
        time.sleep(0.03)

    cap.release()
    st.session_state.detection_running = False
    st.write("Detection stopped.")

# ---------------- Wait for Email Threads to Finish ----------------
for thread in st.session_state.email_threads:
    thread.join()

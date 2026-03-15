import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import time
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import threading

# --- THREAD-SAFE STORAGE ---
# We use a global lock to prevent the app from hanging during data writes
lock = threading.Lock()
if "yamazumi_data" not in st.session_state:
    st.session_state.yamazumi_data = []

class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.prev_rw_x, self.prev_rw_y = 0, 0
        self.prev_lw_x, self.prev_lw_y = 0, 0
        self.last_log_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        category = "NVA"
        action = "Waiting"

        if results.pose_landmarks:
            nose = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
            rw = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            lw = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]

            # Motion calculation
            motion = abs(rw.x - self.prev_rw_x) + abs(rw.y - self.prev_rw_y)
            self.prev_rw_x, self.prev_rw_y = rw.x, rw.y

            if lw.y < nose.y or rw.y < nose.y:
                category, action = "VA", "Process"
            elif motion > 0.002:
                category, action = "VA", "Table Work"
            else:
                category, action = "NVA", "Idle"

        # LOGGING (Thread-Safe)
        curr = time.time()
        if curr - self.last_log_time >= 1.0:
            with lock:
                st.session_state.yamazumi_data.append({"Category": category})
            self.last_log_time = curr

        cv2.putText(img, f"Mode: {category}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI ---
st.title("⏱️ Stable Yamazumi Reporter")

# Main Streamer
webrtc_streamer(
    key="yamazumi-v2",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YamazumiProcessor,
    async_processing=True, # Critical for preventing hangs
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# Analysis Section
if st.session_state.yamazumi_data:
    with lock:
        df = pd.DataFrame(st.session_state.yamazumi_data)
    
    st.subheader("Process Summary")
    counts = df['Category'].value_counts()
    st.bar_chart(counts)

    # PDF Logic
    if st.button("Generate Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Yamazumi Productivity Report", ln=1, align='C')
        for cat, val in counts.items():
            pdf.cell(200, 10, txt=f"{cat}: {val} seconds", ln=1)
        
        pdf_output = pdf.output()
        st.download_button("Download PDF", data=pdf_output, file_name="report.pdf")

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
import queue

# --- 1. GLOBAL QUEUE FOR DATA ---
# Using a Queue is safer than SessionState inside the video loop
result_queue = queue.Queue()

class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.last_log_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        category = "NVA"
        action = "Idle"

        if results.pose_landmarks:
            # Drawing landmarks makes the app heavy; we'll only process logic
            nose = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
            rw = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            lw = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]

            # Simplified logic for stability
            if lw.y < nose.y or rw.y < nose.y:
                category, action = "VA", "High Reach"
            else:
                category, action = "NVA", "Waiting"

        # Push to queue instead of st.session_state
        curr = time.time()
        if curr - self.last_log_time >= 1.0:
            result_queue.put({"Category": category, "Action": action, "Timestamp": time.ctime()})
            self.last_log_time = curr

        cv2.putText(img, f"LVL: {category}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 2. MAIN APP ---
st.title("⏱️ Anti-Freeze Yamazumi Reporter")

if "master_data" not in st.session_state:
    st.session_state.master_data = []

ctx = webrtc_streamer(
    key="stable-yamazumi",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YamazumiProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True, # Keeps UI and Video separate
)

# --- 3. DATA HARVESTER ---
# This pulls data from the queue into the session state safely
while not result_queue.empty():
    st.session_state.master_data.append(result_queue.get())

# --- 4. REPORTING ---
if st.session_state.master_data:
    df = pd.DataFrame(st.session_state.master_data)
    
    st.divider()
    col_chart, col_stat = st.columns(2)
    
    with col_chart:
        st.subheader("VA vs NVA Distribution")
        counts = df['Category'].value_counts()
        st.bar_chart(counts)
    
    with col_stat:
        st.subheader("Key Metrics")
        total = len(df)
        va_count = len(df[df['Category'] == 'VA'])
        ratio = (va_count / total * 100) if total > 0 else 0
        st.metric("Efficiency (VA Ratio)", f"{ratio:.1f}%")
        
        if st.button("Reset Data"):
            st.session_state.master_data = []
            st.rerun()

    # PDF Logic
    if st.button("Generate Final PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(190, 10, "Yamazumi Time Study Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(100, 10, f"Total Observation: {len(df)} seconds", ln=True)
        pdf.cell(100, 10, f"VA Ratio: {ratio:.1f}%", ln=True)
        
        pdf_out = pdf.output()
        st.download_button("Download PDF", data=pdf_out, file_name="Yamazumi_Report.pdf")

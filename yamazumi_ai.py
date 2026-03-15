import os
import time
import queue
import cv2
import av
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from fpdf import FPDF

# --- 1. CLOUD PERMISSION BYPASS ---
# Force MediaPipe to use /tmp for its model downloads/cache
os.environ['MEDIAPIPE_BINARY_GRAPH_SUSPEND_INPUT'] = '1'
os.environ['XDG_CACHE_HOME'] = '/tmp'

# --- 2. SESSION STATE SETUP ---
if "yamazumi_queue" not in st.session_state:
    st.session_state.yamazumi_queue = queue.Queue()

if "master_data" not in st.session_state:
    st.session_state.master_data = []

# --- 3. VIDEO PROCESSOR ---
class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        # We initialize MP inside the class to keep it scoped to the processor
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 0 is 'Lite', required for Cloud stability
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_log_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Process Pose
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        category = "NVA"
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Logic: VA if hands are above nose
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[self.mp_pose.PoseLandmark.NOSE].y
            rw_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
            lw_y = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y

            if lw_y < nose_y or rw_y < nose_y:
                category = "VA"

            # Log every 1 second
            if time.time() - self.last_log_time >= 1.0:
                st.session_state.yamazumi_queue.put({
                    "Category": category, 
                    "Time": time.strftime("%H:%M:%S")
                })
                self.last_log_time = time.time()

        cv2.putText(img, f"Status: {category}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if category == "VA" else (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. APP INTERFACE ---
st.set_page_config(page_title="Yamazumi AI", layout="wide")
st.title("⏱️ Industrial Yamazumi AI Analyzer")

col_v, col_r = st.columns([2, 1])

with col_v:
    webrtc_streamer(
        key="yamazumi-v1",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YamazumiProcessor,
        async_processing=True,
        # Standard STUN servers for web connectivity
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col_r:
    st.subheader("📊 Session Report")
    
    if st.button("🔄 Sync Live Data"):
        while not st.session_state.yamazumi_queue.empty():
            st.session_state.master_data.append(st.session_state.yamazumi_queue.get())

    if st.session_state.master_data:
        df = pd.DataFrame(st.session_state.master_data)
        va_count = len(df[df['Category'] == 'VA'])
        total = len(df)
        efficiency = (va_count / total * 100) if total > 0 else 0
        
        st.metric("Efficiency Ratio", f"{efficiency:.1f}%")
        st.bar_chart(df['Category'].value_counts())

        # PDF Export using FPDF2
        if st.button("📥 Download Report PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("helvetica", "B", 16)
            pdf.cell(40, 10, "Yamazumi Time Study Report")
            pdf.ln(20)
            pdf.set_font("helvetica", "", 12)
            pdf.cell(40, 10, f"Total Duration: {total} seconds")
            pdf.ln(10)
            pdf.cell(40, 10, f"VA Efficiency: {efficiency:.1f}%")
            
            # Using output(dest='S') to get bytes for download button
            pdf_output = pdf.output()
            st.download_button(
                label="Confirm Download",
                data=bytes(pdf_output),
                file_name="yamazumi_report.pdf",
                mime="application/pdf"
            )

    if st.button("Clear Session"):
        st.session_state.master_data = []
        st.rerun()

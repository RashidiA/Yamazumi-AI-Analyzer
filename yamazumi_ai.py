import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import time
import pandas as pd
from fpdf import FPDF
import queue
import os

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
# Create a local writable directory to prevent PermissionError on Streamlit Cloud
MODEL_DIR = os.path.join(os.getcwd(), "mediapipe_models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Initialize Session State for data storage
if "yamazumi_queue" not in st.session_state:
    st.session_state.yamazumi_queue = queue.Queue()

if "master_data" not in st.session_state:
    st.session_state.master_data = []

# --- 2. VIDEO PROCESSOR ENGINE ---
class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize Mediapipe Pose with low complexity for stability
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Fastest model, avoids large downloads
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_log_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Process image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        category = "NVA"
        
        if results.pose_landmarks:
            # Draw skeleton for visual confirmation
            self.mp_draw.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            
            # Key Landmarks
            nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            rw = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            lw = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]

            # VA Logic: Hands above nose level
            if lw.y < nose.y or rw.y < nose.y:
                category = "VA"
            else:
                category = "NVA"

            # Log data every 1 second to the Queue
            curr_time = time.time()
            if curr_time - self.last_log_time >= 1.0:
                # Thread-safe push to the queue
                st.session_state.yamazumi_queue.put({
                    "Category": category, 
                    "Time": time.strftime("%H:%M:%S")
                })
                self.last_log_time = curr_time

        # Video Overlay
        color = (0, 255, 0) if category == "VA" else (0, 0, 255)
        cv2.putText(img, f"STUDY: {category}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. USER INTERFACE ---
st.set_page_config(page_title="Yamazumi AI Analyzer", layout="wide")
st.title("⏱️ Yamazumi AI Analyzer")
st.info("Status: Stand in view to begin tracking. VA = Value Added | NVA = Non-Value Added")

col_left, col_right = st.columns([2, 1])

with col_left:
    webrtc_streamer(
        key="yamazumi-v3",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YamazumiProcessor,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col_right:
    st.subheader("📊 Session Controls")
    
    # Sync data from the background thread to the UI
    if st.button("🔄 Sync & Analyze Data", use_container_width=True):
        while not st.session_state.yamazumi_queue.empty():
            st.session_state.master_data.append(st.session_state.yamazumi_queue.get())
    
    if st.session_state.master_data:
        df = pd.DataFrame(st.session_state.master_data)
        
        # Summary Metrics
        total_s = len(df)
        va_s = len(df[df['Category'] == 'VA'])
        nva_s = total_s - va_s
        va_ratio = (va_s / total_s * 100) if total_s > 0 else 0
        
        st.metric("Efficiency (VA Ratio)", f"{va_ratio:.1f}%")
        st.bar_chart(df['Category'].value_counts())
        
        # --- PDF REPORT GENERATION ---
        if st.button("📄 Generate PDF Report", use_container_width=True):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(190, 10, "Yamazumi Time Study Report", ln=True, align='C')
            pdf.ln(10)
            
            pdf.set_font("Arial", size=12)
            pdf.cell(100, 10, f"Total Observation Time: {total_s} seconds", ln=True)
            pdf.cell(100, 10, f"Value Added (VA) Time: {va_s} seconds", ln=True)
            pdf.cell(100, 10, f"Non-Value Added (NVA) Time: {nva_s} seconds", ln=True)
            pdf.cell(100, 10, f"Calculated Efficiency: {va_ratio:.1f}%", ln=True)
            
            report_out = pdf.output()
            st.download_button(
                label="📥 Download Report",
                data=report_out,
                file_name="Yamazumi_Study_Report.pdf",
                mime="application/pdf"
            )

    if st.button("🗑️ Reset Session", type="secondary"):
        st.session_state.master_data = []
        st.session_state.yamazumi_queue = queue.Queue()
        st.rerun()

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import time
import pandas as pd
from fpdf import FPDF
import io
import queue
import os

# --- 1. PERMISSION & DIRECTORY SETUP ---
# Create a local writable directory for MediaPipe models to avoid PermissionError
MODEL_DIR = os.path.join(os.getcwd(), "mediapipe_models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Global queue to pass data from the video thread to the UI thread safely
if "yamazumi_queue" not in st.session_state:
    st.session_state.yamazumi_queue = queue.Queue()

if "session_history" not in st.session_state:
    st.session_state.session_history = []

# --- 2. VIDEO PROCESSOR ENGINE ---
class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize Pose within the class to keep it isolated
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 0 is fastest and avoids permission issues during download
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_log_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Process image for pose detection
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        category = "NVA"
        
        if results.pose_landmarks:
            # Draw a light skeleton to confirm Abu is watching
            self.mp_draw.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            
            # Identify key landmarks
            nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            rw = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            lw = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]

            # Logic: VA if hands are above nose level (High Reach Assembly)
            if lw.y < nose.y or rw.y < nose.y:
                category = "VA"
            else:
                category = "NVA"

            # Push to queue every 1 second (Avoids st.session_state which causes hangs)
            curr_time = time.time()
            if curr_time - self.last_log_time >= 1.0:
                st.session_state.yamazumi_queue.put({
                    "Category": category, 
                    "Timestamp": time.strftime("%H:%M:%S")
                })
                self.last_log_time = curr_time

        # Visual Feedback on the video stream
        color = (0, 255, 0) if category == "VA" else (0, 0, 255)
        cv2.putText(img, f"LVL: {category}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Yamazumi AI Analyzer", layout="wide")
st.title("⏱️ Yamazumi AI Analyzer v2026")
st.write("Detecting Value-Added (VA) vs Non-Value-Added (NVA) work via Pose Estimation.")

col_video, col_report = st.columns([2, 1])

with col_video:
    st.subheader("Live Feed")
    # Streamer configuration optimized for Streamlit Cloud
    webrtc_streamer(
        key="yamazumi-pro",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YamazumiProcessor,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col_report:
    st.subheader("📊 Session Control")
    # Sync button pulls data from the Queue into the Session History
    if st.button("🔄 Sync & View Report", use_container_width=True):
        while not st.session_state.yamazumi_queue.empty():
            st.session_state.session_history.append(st.session_state.yamazumi_queue.get())
        
        if not st.session_state.session_history:
            st.warning("No data synced yet. Stand in front of the camera!")

    if st.session_state.session_history:
        df = pd.DataFrame(st.session_state.session_history)
        
        # Stats Calculation
        total_sec = len(df)
        va_sec = len(df[df['Category'] == 'VA'])
        va_ratio = (va_sec / total_sec * 100) if total_sec > 0 else 0
        
        st.metric("Efficiency (VA %)", f"{va_ratio:.1f}%")
        st.bar_chart(df['Category'].value_counts())
        
        # PDF Generation Logic
        if st.button("📄 Download PDF Report", use_container_width=True):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(190, 10, "Yamazumi Productivity Report", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.cell(100, 10, f"Total Duration: {total_sec}s", ln=True)
            pdf.cell(100, 10, f"VA Time: {va_sec}s", ln=True)
            pdf.cell(100, 10, f"NVA Time: {total_sec - va_sec}s", ln=True)
            pdf.cell(100, 10, f"VA Efficiency Ratio: {va_ratio:.1f}%", ln=True)
            
            pdf_bytes = pdf.output()
            st.download_button("Click to Download", data=pdf_bytes, file_name="Yamazumi_Report.pdf")

    if st.button("🗑️ Clear Data", type="secondary"):
        st.session_state.session_history = []
        st.session_state.yamazumi_queue = queue.Queue()
        st.rerun()

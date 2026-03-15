import os
import time
import queue
import urllib.request
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# --- 1. SAFE IMPORTS ---
try:
    import cv2
    import av
    import mediapipe as mp
    from fpdf import FPDF
except ImportError as e:
    st.error(f"Critical Dependency Missing: {e}. Please check requirements.txt and reboot.")
    st.stop()

# --- 2. MODEL ASSET SETUP (The PermissionError Fix) ---
# We download the model to /tmp because it's the only writable folder on Streamlit Cloud
MODEL_DIR = "/tmp/mediapipe_models"
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker.task")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

@st.cache_resource
def initialize_model():
    if not os.path.exists(MODEL_PATH):
        MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

# --- 3. SESSION STATE ---
if "yamazumi_queue" not in st.session_state:
    st.session_state.yamazumi_queue = queue.Queue()
if "master_data" not in st.session_state:
    st.session_state.master_data = []

# --- 4. THE AI PROCESSOR ---
class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        self.model_path = initialize_model()
        # Initialize Pose Landmarker using the modern Tasks API
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.last_log_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Convert to MediaPipe image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(mp_image, int(time.time() * 1000))
        
        category = "NVA" # Non-Value Add (Default)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            # Hand indices: Left=15, Right=16 | Nose index: 0
            # Logic: If either hand is higher than the nose (smaller Y), it's Value-Add (VA)
            if landmarks[15].y < landmarks[0].y or landmarks[16].y < landmarks[0].y:
                category = "VA"

            # Log data every 1 second
            if time.time() - self.last_log_time >= 1.0:
                st.session_state.yamazumi_queue.put({
                    "Category": category, 
                    "Time": time.strftime("%H:%M:%S")
                })
                self.last_log_time = time.time()

        # Draw UI Overlay
        color = (0, 255, 0) if category == "VA" else (0, 0, 255)
        cv2.putText(img, f"Status: {category}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="Yamazumi AI Analyzer", layout="wide")
st.title("⏱️ Industrial Yamazumi AI")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Live Analysis")
    webrtc_streamer(
        key="yamazumi-final",
        mode=WebRtcMode.SENDRECV,
        # STUN servers fix the "Connection taking longer than expected" hang
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=YamazumiProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_right:
    st.subheader("📊 Statistics")
    if st.button("🔄 Sync Live Data", use_container_width=True):
        while not st.session_state.yamazumi_queue.empty():
            st.session_state.master_data.append(st.session_state.yamazumi_queue.get())

    if st.session_state.master_data:
        df = pd.DataFrame(st.session_state.master_data)
        va_count = len(df[df['Category'] == 'VA'])
        total = len(df)
        efficiency = (va_count / total * 100) if total > 0 else 0
        
        st.metric("Process Efficiency", f"{efficiency:.1f}%")
        st.bar_chart(df['Category'].value_counts())
        
        if st.button("📥 Download PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(40, 10, f"Yamazumi Report - Efficiency: {efficiency:.1f}%")
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("Download Now", data=pdf_bytes, file_name="Report.pdf")

    if st.button("Clear History"):
        st.session_state.master_data = []
        st.rerun()

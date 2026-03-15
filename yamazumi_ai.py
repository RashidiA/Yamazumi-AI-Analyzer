import os
import time
import queue
import urllib.request
import cv2
import av
import pandas as pd
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from fpdf import FPDF

# --- 1. MODEL ASSET PREPARATION ---
MODEL_PATH = "/tmp/pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

try:
    actual_model_path = download_model()
except Exception as e:
    st.error(f"Model Download Failed: {e}")
    actual_model_path = None

# --- 2. GLOBAL DATA QUEUE ---
if "yamazumi_queue" not in st.session_state:
    st.session_state.yamazumi_queue = queue.Queue()
if "master_data" not in st.session_state:
    st.session_state.master_data = []

# --- 3. OPTIMIZED AI PROCESSOR ---
class YamazumiProcessor(VideoProcessorBase):
    def __init__(self, model_path):
        self.model_path = model_path
        self.landmarker = None
        self.last_log_time = time.time()
        
        # Initialize Landmarker inside the worker thread to prevent UI hang
        if self.model_path:
            try:
                BaseOptions = mp.tasks.BaseOptions
                PoseLandmarker = mp.tasks.vision.PoseLandmarker
                PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=self.model_path),
                    running_mode=VisionRunningMode.VIDEO,
                    min_pose_detection_confidence=0.5
                )
                self.landmarker = PoseLandmarker.create_from_options(options)
            except Exception as e:
                print(f"AI Initialization Error: {e}")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        category = "NO DATA"
        
        if self.landmarker:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                result = self.landmarker.detect_for_video(mp_image, int(time.time() * 1000))
                
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    # Index 0=Nose, 15=L-Wrist, 16=R-Wrist
                    nose_y = landmarks[0].y
                    lw_y = landmarks[15].y
                    rw_y = landmarks[16].y

                    category = "VA" if (lw_y < nose_y or rw_y < nose_y) else "NVA"

                    # Thread-safe logging
                    if time.time() - self.last_log_time >= 1.0:
                        st.session_state.yamazumi_queue.put({
                            "Category": category, 
                            "Time": time.strftime("%H:%M:%S")
                        })
                        self.last_log_time = time.time()
            except:
                category = "ERROR"

        # On-screen HUD
        color = (0, 255, 0) if category == "VA" else (0, 0, 255)
        cv2.putText(img, f"LVL: {category}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. STREAMLIT INTERFACE ---
st.set_page_config(page_title="Yamazumi AI Stable", layout="wide")
st.title("⏱️ Yamazumi AI Analyzer")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Live Feed")
    webrtc_streamer(
        key="yamazumi-final-v7",
        mode=WebRtcMode.SENDRECV,
        # STUN servers help bypass Cloud firewalls
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        },
        video_processor_factory=lambda: YamazumiProcessor(actual_model_path),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_right:
    st.subheader("📊 Session Control")
    if st.button("🔄 Sync Live Data", use_container_width=True):
        while not st.session_state.yamazumi_queue.empty():
            st.session_state.master_data.append(st.session_state.yamazumi_queue.get())

    if st.session_state.master_data:
        df = pd.DataFrame(st.session_state.master_data)
        va_count = len(df[df['Category'] == 'VA'])
        total = len(df)
        ratio = (va_count / total * 100) if total > 0 else 0
        
        st.metric("Work Efficiency (VA)", f"{ratio:.1f}%")
        st.bar_chart(df['Category'].value_counts())
        
        if st.button("📥 Export PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(0, 10, "Yamazumi Productivity Report", ln=True)
            pdf.set_font("Helvetica", size=12)
            pdf.cell(0, 10, f"VA Ratio: {ratio:.1f}%", ln=True)
            pdf_bytes = pdf.output()
            st.download_button("Download PDF", data=bytes(pdf_bytes), file_name="Report.pdf")

    if st.button("Clear Session"):
        st.session_state.master_data = []
        st.rerun()

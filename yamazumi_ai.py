import os
import time
import queue
import urllib.request
import cv2
import av
import pandas as pd
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from fpdf import FPDF

# --- 1. DOWNLOAD MODEL MANUALLY (The Cloud Fix) ---
MODEL_PATH = "/tmp/pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Initializing AI Model for the first time..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# --- 2. DATA STORAGE ---
if "yamazumi_queue" not in st.session_state:
    st.session_state.yamazumi_queue = queue.Queue()
if "master_data" not in st.session_state:
    st.session_state.master_data = []

# --- 3. VISION API PROCESSOR ---
class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        # Using the NEW Vision Tasks API to avoid read-only errors
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.last_log_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Prepare MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(time.time() * 1000)
        
        # Detect
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        category = "NVA"
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            # Indices: Nose=0, Left Wrist=15, Right Wrist=16
            nose_y = landmarks[0].y
            lw_y = landmarks[15].y
            rw_y = landmarks[16].y

            category = "VA" if (lw_y < nose_y or rw_y < nose_y) else "NVA"

            if time.time() - self.last_log_time >= 1.0:
                st.session_state.yamazumi_queue.put({
                    "Category": category, 
                    "Time": time.strftime("%H:%M:%S")
                })
                self.last_log_time = time.time()

        # Visual Feedback
        color = (0, 255, 0) if category == "VA" else (0, 0, 255)
        cv2.putText(img, f"LVL: {category}", (20, 50), 1, 2, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Yamazumi AI", layout="wide")
st.title("⏱️ Yamazumi AI Analyzer")

col_left, col_right = st.columns([2, 1])

with col_left:
    webrtc_streamer(
        key="yamazumi-vision-task",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YamazumiProcessor,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col_right:
    st.subheader("📊 Session Control")
    if st.button("🔄 Sync Live Data"):
        while not st.session_state.yamazumi_queue.empty():
            st.session_state.master_data.append(st.session_state.yamazumi_queue.get())

    if st.session_state.master_data:
        df = pd.DataFrame(st.session_state.master_data)
        va_s = len(df[df['Category'] == 'VA'])
        total = len(df)
        ratio = (va_s / total * 100) if total > 0 else 0
        
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

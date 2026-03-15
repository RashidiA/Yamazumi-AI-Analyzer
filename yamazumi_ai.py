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

# --- 1. MODEL SETUP ---
MODEL_PATH = "/tmp/pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# --- 2. WEBRTC STUN CONFIG (Fixes the hanging screen) ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- 3. SESSION STATE ---
if "yamazumi_queue" not in st.session_state:
    st.session_state.yamazumi_queue = queue.Queue()
if "master_data" not in st.session_state:
    st.session_state.master_data = []

# --- 4. AI PROCESSOR ---
class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        # Using the Vision Tasks API for Cloud Stability
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.last_log_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(mp_image, int(time.time() * 1000))
        
        category = "NVA"
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            # Hands vs Nose logic
            category = "VA" if (landmarks[15].y < landmarks[0].y or landmarks[16].y < landmarks[0].y) else "NVA"

            if time.time() - self.last_log_time >= 1.0:
                st.session_state.yamazumi_queue.put({"Category": category, "Time": time.strftime("%H:%M:%S")})
                self.last_log_time = time.time()

        color = (0, 255, 0) if category == "VA" else (0, 0, 255)
        cv2.putText(img, f"LVL: {category}", (20, 50), 1, 2, color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. APP UI ---
st.set_page_config(page_title="Yamazumi AI", layout="wide")
st.title("⏱️ Yamazumi AI Analyzer")

col_v, col_r = st.columns([2, 1])

with col_v:
    webrtc_streamer(
        key="yamazumi-fixed",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION, # This line fixes the 'hanging'
        video_processor_factory=YamazumiProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_r:
    st.subheader("📊 Session Control")
    if st.button("🔄 Sync Live Data", use_container_width=True):
        while not st.session_state.yamazumi_queue.empty():
            st.session_state.master_data.append(st.session_state.yamazumi_queue.get())

    if st.session_state.master_data:
        df = pd.DataFrame(st.session_state.master_data)
        st.metric("Efficiency", f"{(len(df[df['Category'] == 'VA']) / len(df) * 100):.1f}%")
        st.bar_chart(df['Category'].value_counts())
        
    if st.button("Clear Session"):
        st.session_state.master_data = []
        st.rerun()

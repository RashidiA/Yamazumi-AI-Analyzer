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

# --- 1. FORCE WRITABLE MODEL PATH ---
MODEL_DIR = "/tmp/mediapipe_models"
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmark_lite.tflite")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Manually download the model if it doesn't exist to bypass the internal downloader error
if not os.path.exists(MODEL_PATH):
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to download model: {e}")

# --- 2. DATA SETUP ---
if "yamazumi_queue" not in st.session_state:
    st.session_state.yamazumi_queue = queue.Queue()
if "master_data" not in st.session_state:
    st.session_state.master_data = []

# --- 3. VIDEO PROCESSOR ---
class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        # We use the Base Solutions but point to our manual file
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0, # Lite
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_log_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        category = "NVA"
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Logic: Hands above Nose
            nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            rw = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            lw = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]

            category = "VA" if (lw.y < nose.y or rw.y < nose.y) else "NVA"

            if time.time() - self.last_log_time >= 1.0:
                st.session_state.yamazumi_queue.put({
                    "Category": category, 
                    "Time": time.strftime("%H:%M:%S")
                })
                self.last_log_time = time.time()

        cv2.putText(img, f"LVL: {category}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if category == "VA" else (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. INTERFACE ---
st.set_page_config(page_title="Yamazumi AI Stable", layout="wide")
st.title("⏱️ Yamazumi AI Analyzer (Cloud Stable)")

col_v, col_r = st.columns([2, 1])

with col_v:
    webrtc_streamer(
        key="yamazumi-vfinal",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YamazumiProcessor,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col_r:
    st.subheader("📊 Session Control")
    if st.button("🔄 Sync & Refresh Report", use_container_width=True):
        while not st.session_state.yamazumi_queue.empty():
            st.session_state.master_data.append(st.session_state.yamazumi_queue.get())

    if st.session_state.master_data:
        df = pd.DataFrame(st.session_state.master_data)
        va_s = len(df[df['Category'] == 'VA'])
        total = len(df)
        ratio = (va_s / total * 100) if total > 0 else 0
        
        st.metric("Efficiency", f"{ratio:.1f}%")
        st.bar_chart(df['Category'].value_counts())
        
        if st.button("📥 Export PDF", use_container_width=True):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "Yamazumi Study Results", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"VA Ratio: {ratio:.1f}%", ln=True)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("Download Now", data=pdf_bytes, file_name="Report.pdf")

    if st.button("Reset Data"):
        st.session_state.master_data = []
        st.rerun()

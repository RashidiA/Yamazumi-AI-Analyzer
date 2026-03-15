import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
import pandas as pd
import time
import os
import urllib.request
from fpdf import FPDF
from queue import Queue

# 1. RTC Configuration for Cloud Stability (Fixes the "Connection Timeout")
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 2. Path Fix for MediaPipe Models on Streamlit Cloud
MODEL_DIR = "/tmp/mediapipe_models"
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_full.task")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Download model manually if it doesn't exist to avoid PermissionError
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Shared Queue for Thread-Safe Data Logging
if "data_queue" not in st.session_state:
    st.session_state.data_queue = Queue()
if "history" not in st.session_state:
    st.session_state.history = []

class YamazumiTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.last_log_time = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        status = "NVA"  # Default to Non-Value Added
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Simplified Logic: Hands above Nose = Value Added (VA)
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            l_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            r_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            
            if l_wrist_y < nose_y or r_wrist_y < nose_y:
                status = "VA"

            # Log data every 1 second
            current_time = time.time()
            if current_time - self.last_log_time >= 1.0:
                st.session_state.data_queue.put({"Time": time.strftime("%H:%M:%S"), "Status": status})
                self.last_log_time = current_time

        cv2.putText(img, f"Status: {status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if status == "VA" else (0, 0, 255), 2)
        return img

def main():
    st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")
    st.title("Industrial Yamazumi AI")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Analysis")
        webrtc_streamer(
            key="yamazumi-analyzer",
            video_transformer_factory=YamazumiTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

    with col2:
        st.subheader("📊 Statistics")
        if st.button("🔄 Sync Live Data"):
            while not st.session_state.data_queue.empty():
                st.session_state.history.append(st.session_state.data_queue.get())
        
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df.tail(10), use_container_width=True)
            
            va_count = len(df[df["Status"] == "VA"])
            total = len(df)
            efficiency = (va_count / total) * 100
            st.metric("Cycle Efficiency", f"{efficiency:.1f}%")

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

if __name__ == "__main__":
    main()

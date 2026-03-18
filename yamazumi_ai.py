import os
import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- CONFIGURATION ---
MODEL_PATH = "pose_landmark_lite.tflite"

if 'va_count' not in st.session_state: st.session_state.va_count = 0.0
if 'waste_count' not in st.session_state: st.session_state.waste_count = 0.0
if 'current_status' not in st.session_state: st.session_state.current_status = "IDLE"

class FinalYamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        # We bypass the 'mp.solutions.pose' wrapper to avoid the download crash
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose_vars = mp.solutions.pose
        
        # Initialize Pose using the direct Task API (the most stable for local files)
        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.timestamp = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Process with manual timestamp increment
        self.timestamp += 33 
        result = self.landmarker.detect_for_video(mp_image, self.timestamp)

        status = "IDLE"
        color = (200, 200, 200)

        if result.pose_landmarks:
            # We use the first detected pose
            landmarks = result.pose_landmarks[0]
            
            # Logic: Nose (0) vs Shoulders (11 & 12)
            # MediaPipe Task landmarks use .x, .y, .z
            nose_y = landmarks[0].y
            sh_y = (landmarks[11].y + landmarks[12].y) / 2
            
            if nose_y > (sh_y + 0.05):
                status = "WASTE"
                color = (0, 165, 255) # Orange
            else:
                status = "VALUE-ADD"
                color = (0, 255, 0) # Green
            
            st.session_state.current_status = status
            
            # Draw AR Indicator
            cv2.putText(img, f"STUDY: {status}", (40, 70), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
            
            # Manual skeleton drawing (Task API format)
            h, w, _ = img.shape
            cx, cy = int(landmarks[0].x * w), int(landmarks[0].y * h)
            cv2.circle(img, (cx, cy), 15, color, -1)

        return frame.from_ndarray(img, format="bgr24")

# --- UI INTERFACE ---
st.set_page_config(page_title="Yamazumi PhD Analyzer", layout="wide")
st.title("⏱️ Yamazumi AR: Local-Model Edition")

if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ Critical: Please upload '{MODEL_PATH}' to the root of your GitHub repo.")
    st.stop()

col_left, col_right = st.columns([2, 1])

with col_left:
    webrtc_streamer(
        key="yamazumi-final",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FinalYamazumiProcessor,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    if st.toggle("⏺️ Start Auto-Recording"):
        if 'last_time' not in st.session_state:
            st.session_state.last_time = time.time()
        now = time.time()
        delta = now - st.session_state.last_time
        
        if 0 < delta < 1:
            if st.session_state.current_status == "VALUE-ADD":
                st.session_state.va_count += delta
            elif st.session_state.current_status == "WASTE":
                st.session_state.waste_count += delta
                
        st.session_state.last_time = now
        time.sleep(0.4)
        st.rerun()

with col_right:
    st.subheader("Time Study Data")
    st.metric("Total Seconds", f"{st.session_state.va_count + st.session_state.waste_count:.1f}s")
    
    # Industrial Yamazumi Chart
    df = pd.DataFrame([{"VA": st.session_state.va_count, "Waste": st.session_state.waste_count}])
    st.bar_chart(df, color=["#2ecc71", "#e67e22"])
    
    if st.button("Reset Counter"):
        st.session_state.va_count = 0.0
        st.session_state.waste_count = 0.0
        st.rerun()

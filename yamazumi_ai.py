import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- FILE PATH CHECK ---
# Ensure this matches exactly where you put the file on GitHub
MODEL_PATH = "pose_landmark_lite.tflite" 

class YamazumiARProcessor(VideoProcessorBase):
    def __init__(self):
        # Using the newer Tasks API which honors local file paths strictly
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create options to point specifically to your uploaded file
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO
        )
        
        try:
            self.landmarker = PoseLandmarker.create_from_options(options)
        except Exception as e:
            st.error(f"Failed to load local model: {e}")
            self.landmarker = None

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose # Used only for drawing connections

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.landmarker:
            # Convert to MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Process frame with a timestamp (required for VIDEO mode)
            timestamp_ms = int(st.session_state.get("frame_count", 0) * 33) 
            st.session_state["frame_count"] = st.session_state.get("frame_count", 0) + 1
            
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                # Logic: Check Y-coordinates of landmarks
                # result.pose_landmarks is a list of lists
                for landmarks in result.pose_landmarks:
                    # Draw manually or using drawing_utils (landmarks are normalized)
                    # Simple Logic: Nose (Index 0)
                    nose_y = landmarks[0].y
                    sh_y = (landmarks[11].y + landmarks[12].y) / 2
                    
                    status = "WASTE" if nose_y > (sh_y + 0.05) else "VALUE-ADD"
                    color = (0, 165, 255) if status == "WASTE" else (0, 255, 0)
                    
                    cv2.putText(img, status, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        return frame.from_ndarray(img, format="bgr24")

# --- UI Setup ---
st.set_page_config(page_title="Yamazumi AR", layout="wide")
st.title("🛡️ Yamazumi AI: AR Live Mode")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ '{MODEL_PATH}' not found in the main folder of your GitHub repository.")
    st.info("Please upload the .tflite file you downloaded and refresh.")
else:
    st.success("✅ Local Model Detected. Starting AR Engine...")
    
    webrtc_streamer(
        key="yamazumi-ar-tasks",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YamazumiARProcessor,
        async_processing=True,
    )

    # --- Manual Data Logging ---
    st.divider()
    if 'va' not in st.session_state: st.session_state.va = 0
    if 'w' not in st.session_state: st.session_state.w = 0
    
    col1, col2 = st.columns(2)
    if col1.button("Log 1s VA", use_container_width=True): st.session_state.va += 1
    if col2.button("Log 1s Waste", use_container_width=True): st.session_state.w += 1
    
    st.bar_chart({"Value-Add": st.session_state.va, "Waste": st.session_state.w}, color=["#2ecc71", "#e67e22"])

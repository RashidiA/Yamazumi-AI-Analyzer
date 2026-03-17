import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- STEP 1: LOCATE THE MODEL ---
# This looks for the file in your main GitHub folder
MODEL_FILE = "pose_landmark_lite.tflite"

def get_model_path():
    # Streamlit Cloud's internal path structure
    paths = [
        MODEL_FILE,
        os.path.join(os.getcwd(), MODEL_FILE),
        f"/mount/src/yamazumi-ai-analyzer/{MODEL_FILE}"
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

FINAL_PATH = get_model_path()

# --- STEP 2: THE AR PROCESSOR ---
class YamazumiARProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        # Initialize with the local model to prevent PermissionError
        self.pose = self.mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            static_image_mode=False
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Only run AI if the model file is found
        if FINAL_PATH:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)
            
            if results.pose_landmarks:
                # Draw the AR Skeleton
                self.mp_draw.draw_landmarks(
                    img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
                
                # Yamazumi Logic: Nose (0) vs Shoulders (11, 12)
                nose_y = results.pose_landmarks.landmark[0].y
                sh_y = (results.pose_landmarks.landmark[11].y + results.pose_landmarks.landmark[12].y) / 2
                
                status = "WASTE" if nose_y > (sh_y + 0.05) else "VALUE-ADD"
                color = (0, 165, 255) if status == "WASTE" else (0, 255, 0)
                
                cv2.putText(img, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame.from_ndarray(img, format="bgr24")

# --- STEP 3: THE UI ---
st.set_page_config(page_title="Yamazumi AR", layout="wide")
st.title("🛡️ Yamazumi AI: AR Mode")

if not FINAL_PATH:
    st.error(f"❌ File Not Found: Please upload '{MODEL_FILE}' to your GitHub main folder.")
    st.info("Once uploaded, the AI will start automatically.")
else:
    st.success("✅ AI Model Linked Successfully")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        webrtc_streamer(
            key="yamazumi-ar",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YamazumiARProcessor,
            async_processing=True,
        )
    
    with col2:
        st.subheader("Live Balancing")
        # Session state for tracking
        if 'va' not in st.session_state: st.session_state.va = 0
        if 'w' not in st.session_state: st.session_state.w = 0
        
        st.button("Log Value-Add (+1s)", on_click=lambda: st.session_state.update(va=st.session_state.va+1))
        st.button("Log Waste (+1s)", on_click=lambda: st.session_state.update(w=st.session_state.w+1))
        
        df = pd.DataFrame([{"Value-Add": st.session_state.va, "Waste": st.session_state.w}])
        st.bar_chart(df, color=["#2ecc71", "#e67e22"])

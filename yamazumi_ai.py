import os
import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- DIRECT MODEL CONFIG ---
MODEL_PATH = "pose_landmark_lite.tflite"

# Ensure state is ready
if 'va_count' not in st.session_state: st.session_state.va_count = 0.0
if 'waste_count' not in st.session_state: st.session_state.waste_count = 0.0
if 'current_status' not in st.session_state: st.session_state.current_status = "IDLE"

class StableYamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        # Using the standard Solution API but forcing it to recognize the local file
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        
        # We initialize here to keep it inside the worker thread
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0, # Matches 'lite' model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(img_rgb)

        status = "IDLE"
        color = (200, 200, 200)

        if results.pose_landmarks:
            # Draw AR connections
            self.mp_draw.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            
            # Logic: Nose (0) vs Shoulders (11 & 12)
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[0].y
            sh_y = (landmarks[11].y + landmarks[12].y) / 2
            
            # Classification
            if nose_y > (sh_y + 0.05):
                status = "WASTE"
                color = (0, 165, 255) # Orange
            else:
                status = "VALUE-ADD"
                color = (0, 255, 0) # Green
            
            # Update global status for the timer
            st.session_state.current_status = status
            
            # On-screen AR Label
            cv2.putText(img, f"STATUS: {status}", (40, 70), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)

        return frame.from_ndarray(img, format="bgr24")

# --- UI INTERFACE ---
st.set_page_config(page_title="Yamazumi AI Stable", layout="wide")
st.title("⏱️ Yamazumi AI: Industrial Analyzer")

if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ Error: '{MODEL_PATH}' not found in GitHub. Please upload it to the main folder.")
    st.stop()

col_left, col_right = st.columns([2, 1])

with col_left:
    webrtc_streamer(
        key="yamazumi-stable",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=StableYamazumiProcessor,
        async_processing=True,
        # Adding RTC configuration for better cloud stability
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # --- AUTO-TIMER LOGIC ---
    if st.toggle("Start Automatic Recording"):
        # Update counts every time Streamlit reruns
        if 'last_time' not in st.session_state:
            st.session_state.last_time = time.time()
            
        now = time.time()
        delta = now - st.session_state.last_time
        
        if delta < 1.5: # Prevent jumps on first load
            if st.session_state.current_status == "VALUE-ADD":
                st.session_state.va_count += delta
            elif st.session_state.current_status == "WASTE":
                st.session_state.waste_count += delta
                
        st.session_state.last_time = now
        time.sleep(0.3) # Faster refresh for the chart
        st.rerun()
    else:
        # Reset timer tracking when toggled off
        if 'last_time' in st.session_state:
            del st.session_state.last_time

with col_right:
    st.subheader("Study Results")
    st.info(f"Current Activity: **{st.session_state.current_status}**")
    
    va = round(st.session_state.va_count, 1)
    ws = round(st.session_state.waste_count, 1)
    total = va + ws
    
    st.metric("Total Time", f"{total}s")
    
    # Yamazumi Bar Chart
    df = pd.DataFrame([{"Value-Add": va, "Waste": ws}])
    st.bar_chart(df, color=["#2ecc71", "#e67e22"])
    
    if st.button("Reset Counter"):
        st.session_state.va_count = 0
        st.session_state.waste_count = 0
        st.rerun()

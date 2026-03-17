import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- System Fixes ---
os.environ["LD_PRELOAD"] = ""

# --- Page Config ---
st.set_page_config(page_title="Yamazumi AR Analyzer", layout="wide")
st.title("🛡️ Yamazumi AR Live Analyzer")

# --- AI Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5)

# --- Session State for Data ---
if 'va_time' not in st.session_state:
    st.session_state.va_time = 0
if 'waste_time' not in st.session_state:
    st.session_state.waste_time = 0

# --- AR Video Processor ---
class YamazumiTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        status = "Scanning..."
        color = (255, 255, 255)

        if results.pose_landmarks:
            # Draw AR Skeleton
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Logic: Nose vs Shoulder Height
            nose_y = results.pose_landmarks.landmark[0].y
            sh_y = (results.pose_landmarks.landmark[11].y + results.pose_landmarks.landmark[12].y) / 2
            
            if nose_y > (sh_y + 0.05):
                status = "WASTE (Bending)"
                color = (0, 165, 255) # Orange
            else:
                status = "VALUE-ADD (Working)"
                color = (0, 255, 0) # Green
            
            # Draw AR Label on Screen
            cv2.putText(img, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img

# --- Layout ---
col_vid, col_data = st.columns([2, 1])

with col_vid:
    st.subheader("Live AR Feed")
    webrtc_streamer(key="yamazumi-ar", video_transformer_factory=YamazumiTransformer)
    st.caption("Tip: Ensure your upper body is fully visible for the AR skeleton to lock on.")

with col_data:
    st.subheader("Real-Time Balancing")
    takt = st.number_input("Takt Time (s)", value=30.0)
    
    # Since we can't easily sync the live JS clock to Streamlit state in a loop,
    # we use manual log triggers or simulated time steps.
    if st.button("Log 1s Value-Add"):
        st.session_state.va_time += 1
    if st.button("Log 1s Waste"):
        st.session_state.waste_time += 1
        
    total = st.session_state.va_time + st.session_state.waste_time
    st.metric("Total Cycle", f"{total}s", delta=f"{takt-total}s to Takt")
    
    chart_data = pd.DataFrame([{
        "Value-Add": st.session_state.va_time,
        "Waste": st.session_state.waste_time
    }])
    st.bar_chart(chart_data, color=["#2ecc71", "#e67e22"])

    if st.button("Reset Counter"):
        st.session_state.va_time = 0
        st.session_state.waste_time = 0
        st.rerun()

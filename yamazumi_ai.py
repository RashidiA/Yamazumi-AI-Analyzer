import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- System Fixes ---
os.environ["LD_PRELOAD"] = ""

st.set_page_config(page_title="Yamazumi AR Analyzer", layout="wide")
st.title("🛡️ Yamazumi AR Live Analyzer")

# --- AR Video Processor Class ---
class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize MediaPipe inside the processor to avoid permission errors
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            model_complexity=0, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(img_rgb)

        status = "Scanning..."
        color = (255, 255, 255)

        if results.pose_landmarks:
            # Draw AR Skeleton Overlay
            self.mp_drawing.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            
            # Yamazumi Logic: Nose Height vs Shoulder Height
            # Landmarks: Nose (0), L-Shoulder (11), R-Shoulder (12)
            nose_y = results.pose_landmarks.landmark[0].y
            sh_y = (results.pose_landmarks.landmark[11].y + results.pose_landmarks.landmark[12].y) / 2
            
            # Classification
            if nose_y > (sh_y + 0.05):
                status = "WASTE (Bending/Searching)"
                color = (0, 165, 255) # Orange in BGR
            else:
                status = "VALUE-ADD (Working)"
                color = (0, 255, 0) # Green in BGR
            
            # Render text on the live AR feed
            cv2.putText(img, status, (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)

        return frame.from_ndarray(img, format="bgr24")

# --- Streamlit UI Layout ---
col_vid, col_data = st.columns([2, 1])

with col_vid:
    st.subheader("Live AR Analysis")
    # Using webrtc_streamer with the updated processor
    webrtc_streamer(
        key="yamazumi-ar-v2",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YamazumiProcessor,
        async_processing=True,
    )
    st.caption("AI Status: Live AR Overlay Active")

with col_data:
    st.subheader("Workload Summary")
    takt = st.number_input("Target Takt Time (s)", value=30.0)
    
    # Session State for manual logging (until we sync the JS clock)
    if 'va_s' not in st.session_state: st.session_state.va_s = 0
    if 'w_s' not in st.session_state: st.session_state.w_s = 0
    
    c1, c2 = st.columns(2)
    if c1.button("Add 1s VA", use_container_width=True): st.session_state.va_s += 1
    if c2.button("Add 1s Waste", use_container_width=True): st.session_state.w_s += 1
        
    total = st.session_state.va_s + st.session_state.w_s
    st.metric("Total Cycle Time", f"{total}s", delta=f"{takt-total}s vs Takt")
    
    # Yamazumi Stacked Chart
    chart_data = pd.DataFrame([{
        "Value-Add": st.session_state.va_s,
        "Waste": st.session_state.w_s
    }])
    st.bar_chart(chart_data, color=["#2ecc71", "#e67e22"])

    if st.sidebar.button("Reset All Data"):
        st.session_state.va_s = 0
        st.session_state.w_s = 0
        st.rerun()

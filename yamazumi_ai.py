import os
import sys

# --- THE "LOOP-BREAKER" FIX ---
# This tells the app to ignore system-level graphics drivers that cause crashes
os.environ["LD_PRELOAD"] = ""
os.environ["OPENCV_VIDEOIO_PRIORITY_BACKEND"] = "0"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from datetime import datetime

# --- Page Setup ---
st.set_page_config(page_title="Yamazumi AI Analyzer", layout="wide")
st.title("⏱️ Yamazumi AI Workload Analyzer")
st.caption("Industrial Engineering Tool | Python 3.11 Stable")

# --- AI Model Initialization ---
@st.cache_resource
def load_pose_engine():
    # model_complexity=0 is the fastest and most stable for cloud servers
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose, mp_pose, mp.solutions.drawing_utils

try:
    pose_engine, mp_pose, mp_draw = load_pose_engine()
except Exception as e:
    st.error(f"AI Engine failed to start: {e}")
    st.stop()

# --- State Management ---
if 'yamazumi_data' not in st.session_state:
    st.session_state.yamazumi_data = {"Value-Add": 0.0, "Waste": 0.0}
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Sidebar Controls ---
st.sidebar.header("Study Parameters")
takt_time = st.sidebar.number_input("Takt Time (s)", value=30.0)
step_value = st.sidebar.slider("Time per Capture (s)", 0.1, 1.0, 0.5)

if st.sidebar.button("Reset Study", type="primary"):
    st.session_state.yamazumi_data = {"Value-Add": 0.0, "Waste": 0.0}
    st.session_state.history = []
    st.rerun()

# --- Main Interface ---
col_cam, col_chart = st.columns([1, 1])

with col_cam:
    st.subheader("Action Capture")
    # Camera Input is the most stable way to avoid video stream timeouts
    img_file = st.camera_input("Snapshot of Work Step")

    if img_file:
        # Process Image
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose_engine.process(rgb)
        
        if results.pose_landmarks:
            # Draw for feedback
            mp_draw.draw_landmarks(rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Yamazumi Logic: Head Height vs Shoulder Height
            # (In Mediapipe Y: 0 is top, 1 is bottom)
            nose_y = results.pose_landmarks.landmark[0].y
            shoulder_y = (results.pose_landmarks.landmark[11].y + results.pose_landmarks.landmark[12].y) / 2
            
            # Logic: If nose is significantly lower than shoulders (bending), it's Waste
            category = "Waste" if nose_y > (shoulder_y + 0.03) else "Value-Add"
            
            # Update Data
            st.session_state.yamazumi_data[category] += step_value
            st.session_state.history.append({
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Type": category
            })
            
            st.image(rgb, use_container_width=True)
            st.success(f"Captured: {category}")
        else:
            st.warning("Worker not detected. Please reposition.")

with col_chart:
    st.subheader("Yamazumi Balancing")
    total_time = sum(st.session_state.yamazumi_data.values())
    
    # KPIs
    kpi1, kpi2 = st.columns(2)
    kpi1.metric("Current Cycle", f"{total_time:.1f}s")
    kpi2.metric("Takt Gap", f"{takt_time - total_time:.1f}s")
    
    # The Yamazumi Stacked Bar
    if total_time > 0:
        chart_df = pd.DataFrame([st.session_state.yamazumi_data])
        st.bar_chart(chart_df, color=["#2ecc71", "#e74c3c"]) # Green/Red
        
    # Log Data
    if st.session_state.history:
        st.write("Recent Activity Log")
        st.dataframe(pd.DataFrame(st.session_state.history).tail(5), use_container_width=True)

# Export Function
if total_time > 0:
    csv = pd.DataFrame(st.session_state.history).to_csv(index=False)
    st.download_button("Download Study Data", csv, "yamazumi_report.csv", "text/csv")

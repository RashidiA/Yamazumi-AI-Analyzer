import os
import sys

# --- CRITICAL FIX FOR STREAMLIT CLOUD ---
# This must run before importing cv2 to prevent libgthread errors
os.environ["LD_PRELOAD"] = "" 
try:
    import cv2
except ImportError:
    # Fallback for specific Linux environments
    os.system("apt-get update && apt-get install -y libglib2.0-0")
    import cv2

import streamlit as st
import numpy as np
import pandas as pd
import mediapipe as mp
from datetime import datetime

# --- Page Setup ---
st.set_page_config(
    page_title="Yamazumi AI Analyzer",
    page_icon="⏱️",
    layout="wide"
)

st.title("⏱️ Yamazumi AI Workload Analyzer")
st.markdown("---")

# --- AI Engine Initialization ---
@st.cache_resource
def load_mediapipe():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose, mp_pose, mp.solutions.drawing_utils

pose_engine, mp_pose, mp_draw = load_mediapipe()

# --- Session State Management ---
if 'log_data' not in st.session_state:
    st.session_state.log_data = []
if 'totals' not in st.session_state:
    st.session_state.totals = {"Value-Add": 0.0, "Waste": 0.0}

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
takt_time = st.sidebar.number_input("Target Takt Time (s)", min_value=1.0, value=30.0)
capture_weight = st.sidebar.slider("Time per Capture (s)", 0.1, 2.0, 0.5)

if st.sidebar.button("Reset Analysis", type="primary"):
    st.session_state.log_data = []
    st.session_state.totals = {"Value-Add": 0.0, "Waste": 0.0}
    st.rerun()

# --- Main Layout ---
col_cam, col_stats = st.columns([1.5, 1])

with col_cam:
    st.subheader("🎥 Motion Capture")
    img_file = st.camera_input("Take a snapshot of the work step")

    if img_file:
        # Convert to OpenCV format
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose_engine.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw landmarks for visual feedback
            mp_draw.draw_landmarks(
                rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Logic: Analyze Posture (Nose vs Shoulder line)
            # In MediaPipe, Y increases downward (0 is top, 1 is bottom)
            nose_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
            l_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            r_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            avg_shoulder_y = (l_shoulder_y + r_shoulder_y) / 2
            
            # If nose is significantly lower than shoulders, classify as 'Waste' (bending/searching)
            if nose_y > (avg_shoulder_y + 0.02):
                category = "Waste"
                color = "orange"
            else:
                category = "Value-Add"
                color = "green"
            
            # Update State
            st.session_state.totals[category] += capture_weight
            st.session_state.log_data.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Category": category,
                "Duration": capture_weight
            })
            
            st.image(rgb_frame, use_container_width=True)
            st.success(f"Detected: {category}")
        else:
            st.warning("No person detected. Please adjust the camera.")

with col_stats:
    st.subheader("📊 Yamazumi Stack")
    
    total_cycle = sum(st.session_state.totals.values())
    diff = takt_time - total_cycle
    
    # Metrics
    m1, m2 = st.columns(2)
    m1.metric("Cycle Time", f"{total_cycle:.1f}s")
    m2.metric("Takt Gap", f"{diff:.1f}s", delta=diff, delta_color="inverse")
    
    # Yamazumi Bar Chart
    if total_cycle > 0:
        chart_df = pd.DataFrame([st.session_state.totals])
        st.bar_chart(chart_df, color=["#2ecc71", "#e67e22"]) # Green and Orange
        
    # Log Table
    if st.session_state.log_data:
        st.write("Recent Steps:")
        st.table(pd.DataFrame(st.session_state.log_data).tail(5))

# Export Data
if total_cycle > 0:
    csv = pd.DataFrame(st.session_state.log_data).to_csv(index=False)
    st.download_button("Export Study to CSV", csv, "yamazumi_report.csv", "text/csv")

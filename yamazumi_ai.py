import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")

st.title("⏱️ Industrial Yamazumi AI Analyzer")
st.caption("Motion-Based Workload Balancing & Cycle Time Extraction")

# --- Optimized Model Loading ---
@st.cache_resource
def get_pose_model():
    # Accessing the solutions sub-module directly for environment stability
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose_analyzer = get_pose_model()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Session State ---
if 'cycle_times' not in st.session_state:
    st.session_state.cycle_times = {"VA": 0.0, "NVA": 0.0, "Waste": 0.0}
if 'log' not in st.session_state:
    st.session_state.log = []

# --- Sidebar ---
st.sidebar.header("Industrial Parameters")
takt_time = st.sidebar.number_input("Takt Time (s)", min_value=1.0, value=30.0)
increment = st.sidebar.slider("Time Increment (s)", 0.1, 2.0, 0.5)

if st.sidebar.button("Reset Data"):
    st.session_state.cycle_times = {"VA": 0.0, "NVA": 0.0, "Waste": 0.0}
    st.session_state.log = []
    st.rerun()

# --- Analysis Logic ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🎥 Motion Capture")
    img_file = st.camera_input("Snapshot for Analysis")

    if img_file:
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_analyzer.process(rgb_frame)

        if results.pose_landmarks:
            annotated = rgb_frame.copy()
            mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Ergonomic Logic
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2

            category = "Waste" if nose_y > (shoulder_y + 0.05) else "VA"
            
            st.session_state.cycle_times[category] += increment
            st.session_state.log.append({"Action": category, "Duration": increment})
            
            st.image(annotated, use_container_width=True)
            if category == "Waste":
                st.warning("Ergonomic Waste (Excessive Bending) Detected")
            else:
                st.success("Value-Added Motion Detected")

with col2:
    st.subheader("📊 Yamazumi Analysis")
    total = sum(st.session_state.cycle_times.values())
    st.metric("Total Cycle Time", f"{total:.1f}s", delta=f"{takt_time - total:.1f}s Takt Gap")
    
    st.bar_chart(pd.DataFrame([st.session_state.cycle_times]))
    
    if total > takt_time:
        st.error(f"OVERBURDEN: {total - takt_time:.1f}s over Takt!")

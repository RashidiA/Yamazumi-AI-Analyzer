import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# --- Page Configuration ---
st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")

st.title("⏱️ Industrial Yamazumi AI Analyzer")
st.caption("Motion-Based Workload Balancing & Cycle Time Extraction")

# --- Robust Model Loading ---
@st.cache_resource
def get_pose_engine():
    # Explicitly reaching the Pose class to avoid the 'no attribute solutions' error
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose, mp_pose, mp_drawing

# Initialize components
try:
    pose_model, mp_pose, mp_drawing = get_pose_engine()
except Exception as e:
    st.error(f"AI Engine failed to start: {e}")
    st.stop()

# --- Session State ---
if 'cycle_times' not in st.session_state:
    st.session_state.cycle_times = {"VA": 0.0, "Waste": 0.0}

# --- Sidebar ---
st.sidebar.header("Industrial Parameters")
takt_time = st.sidebar.number_input("Target Takt Time (s)", min_value=1.0, value=30.0)
increment = st.sidebar.slider("Capture Interval (s)", 0.1, 2.0, 0.5)

if st.sidebar.button("Clear Data"):
    st.session_state.cycle_times = {"VA": 0.0, "Waste": 0.0}
    st.rerun()

# --- Main Interface ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🎥 Motion Capture")
    img_file = st.camera_input("Log Motion Point")

    if img_file:
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose_model.process(rgb_frame)

        if results.pose_landmarks:
            # Drawing landmarks for visual feedback
            mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Posture Logic: Bending Detection (Waste)
            # Nose (index 0), Shoulders (index 11/12)
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[0].y
            avg_shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
            
            # Categorize: If nose is significantly below shoulders, it's Waste
            category = "Waste" if nose_y > (avg_shoulder_y + 0.05) else "VA"
            st.session_state.cycle_times[category] += increment
            
            st.image(rgb_frame, use_container_width=True)
            if category == "Waste":
                st.warning("Waste Detected: Operator Bending/Reaching")
            else:
                st.success("Value-Added Motion Recorded")

with col2:
    st.subheader("📊 Yamazumi Balancing")
    total_time = sum(st.session_state.cycle_times.values())
    st.metric("Total Cycle Time", f"{total_time:.1f}s", delta=f"{takt_time - total_time:.1f}s remaining")
    
    # Visualization
    st.bar_chart(pd.DataFrame([st.session_state.cycle_times]))

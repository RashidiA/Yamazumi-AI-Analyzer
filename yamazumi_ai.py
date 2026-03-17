import streamlit as st
import cv2
import numpy as np
import pandas as pd

# Defensive MediaPipe Imports
import mediapipe as mp
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    # Fallback for different environments
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing

# --- Page Config ---
st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")

st.title("⏱️ Industrial Yamazumi AI Analyzer")
st.caption("Motion-Based Workload Balancing & Cycle Time Extraction")

# --- Optimized Model Loading ---
@st.cache_resource
def get_pose_model():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# Initialize Engine
try:
    pose_engine = get_pose_model()
except Exception as e:
    st.error(f"Error initializing AI engine: {e}")
    st.stop()

# --- Session State Management ---
if 'cycle_times' not in st.session_state:
    st.session_state.cycle_times = {"VA": 0.0, "NVA": 0.0, "Waste": 0.0}
if 'log' not in st.session_state:
    st.session_state.log = []

# --- Sidebar Controls ---
st.sidebar.header("Industrial Parameters")
takt_time = st.sidebar.number_input("Target Takt Time (s)", min_value=1.0, value=30.0)
increment_val = st.sidebar.slider("Analysis Time Increment (s)", 0.1, 2.0, 0.5)

if st.sidebar.button("Reset All Data", type="primary"):
    st.session_state.cycle_times = {"VA": 0.0, "NVA": 0.0, "Waste": 0.0}
    st.session_state.log = []
    st.rerun()

# --- Main Interface ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🎥 Motion Capture Analysis")
    img_file = st.camera_input("Capture operator movement")

    if img_file:
        # Convert to OpenCV
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Inference
        results = pose_engine.process(rgb_frame)

        if results.pose_landmarks:
            annotated_image = rgb_frame.copy()
            mp_drawing.draw_landmarks(
                annotated_image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )
            
            # Posture Logic: Detect Bending (Waste)
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2

            if nose_y > shoulder_y + 0.05:
                category = "Waste"
                st.warning("⚠️ Ergonomic Risk: Excessive Bending (Waste)")
            else:
                category = "VA"
                st.success("✅ Standard Motion (Value-Added)")

            # Record Data
            st.session_state.cycle_times[category] += increment_val
            st.session_state.log.append({"Action": category, "Duration": increment_val})
            
            st.image(annotated_image, use_container_width=True)
        else:
            st.error("Operator not detected.")
            st.image(rgb_frame, use_container_width=True)

with col2:
    st.subheader("📊 Yamazumi Analysis")
    total_time = sum(st.session_state.cycle_times.values())
    
    m1, m2 = st.columns(2)
    m1.metric("Total Cycle Time", f"{total_time:.1f}s")
    m2.metric("Takt Gap", f"{takt_time - total_time:.1f}s")

    # Bar Chart
    df_chart = pd.DataFrame([st.session_state.cycle_times])
    st.bar_chart(df_chart)

    if total_time > takt_time:
        st.error(f"OVERBURDEN: Cycle exceeds Takt by {total_time - takt_time:.1f}s")
    
    with st.expander("Detailed Analysis Log"):
        if st.session_state.log:
            df_log = pd.DataFrame(st.session_state.log)
            st.table(df_log)

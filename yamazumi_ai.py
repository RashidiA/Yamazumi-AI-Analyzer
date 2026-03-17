import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# --- Page Configuration ---
st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")

st.title("⏱️ Industrial Yamazumi AI Analyzer")
st.caption("Motion-Based Workload Balancing & Cycle Time Extraction")

# --- AI Model Loading ---
@st.cache_resource
def get_pose_model():
    # Load MediaPipe Pose solution
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# Initialize MediaPipe components
try:
    pose_engine = get_pose_model()
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
except Exception as e:
    st.error(f"AI Engine Initialization Failed: {e}")
    st.stop()

# --- Session State ---
if 'cycle_times' not in st.session_state:
    st.session_state.cycle_times = {"VA": 0.0, "NVA": 0.0, "Waste": 0.0}
if 'log' not in st.session_state:
    st.session_state.log = []

# --- Sidebar ---
st.sidebar.header("Industrial Settings")
takt_time = st.sidebar.number_input("Target Takt Time (s)", min_value=1.0, value=30.0)
increment = st.sidebar.slider("Time Increment per Capture (s)", 0.1, 2.0, 0.5)

if st.sidebar.button("Reset Study Data", type="primary"):
    st.session_state.cycle_times = {"VA": 0.0, "NVA": 0.0, "Waste": 0.0}
    st.session_state.log = []
    st.rerun()

# --- Layout ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🎥 Motion Analysis")
    img_file = st.camera_input("Capture Worker Motion")

    if img_file:
        # Convert image to OpenCV format
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run AI Pose Inference
        results = pose_engine.process(rgb_frame)

        if results.pose_landmarks:
            annotated_image = rgb_frame.copy()
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            
            # Posture Logic for Yamazumi Categorization
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2

            # If nose is lower than shoulders, classify as Waste (Bending)
            category = "Waste" if nose_y > (shoulder_y + 0.05) else "VA"
            
            # Update Data
            st.session_state.cycle_times[category] += increment
            st.session_state.log.append({"Action": category, "Duration": increment})
            
            st.image(annotated_image, use_container_width=True)
            if category == "Waste":
                st.warning("⚠️ Waste Detected: Operator Bending (Ergonomic Risk)")
            else:
                st.success("✅ Value-Added Motion Recorded")
        else:
            st.error("Operator not detected in frame.")

with col2:
    st.subheader("📊 Yamazumi Balancing")
    total_time = sum(st.session_state.cycle_times.values())
    
    st.metric("Total Cycle Time", f"{total_time:.1f}s", 
              delta=f"{takt_time - total_time:.1f}s vs Takt", delta_color="inverse")
    
    # Generate Yamazumi Chart
    df_chart = pd.DataFrame([st.session_state.cycle_times])
    st.bar_chart(df_chart)

    if total_time > takt_time:
        st.error(f"OVERBURDEN: Cycle exceeds Takt by {total_time - takt_time:.1f}s")
    
    with st.expander("Cycle Log"):
        if st.session_state.log:
            st.table(pd.DataFrame(st.session_state.log))

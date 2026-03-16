import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose_module
from mediapipe.python.solutions import drawing_utils as mp_drawing

# --- Page Config ---
st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")
st.title("⏱️ Industrial Yamazumi AI Analyzer")

# --- Optimized Model Loading ---
@st.cache_resource
def get_pose_model():
    return mp_pose_module.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose = get_pose_model()
mp_pose = mp_pose_module
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

# --- Session State ---
if 'cycle_times' not in st.session_state:
    # Example starting data: [Value-Added, Non-Value-Added, Waste]
    st.session_state.cycle_times = {"VA": 10, "NVA": 5, "Waste": 2}

# --- Sidebar ---
st.sidebar.header("Industrial Controls")
if st.sidebar.button("Reset Analysis"):
    st.session_state.cycle_times = {"VA": 0, "NVA": 0, "Waste": 0}
    st.rerun()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Motion Capture")
    img_file = st.camera_input("Capture frame for analysis")

    if img_file:
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw on a copy to keep the original clean if needed
            annotated_image = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec
            )
            st.image(annotated_image, channels="BGR", use_container_width=True)
            st.success("Ergonomic/Motion Data Extracted")
            
            # Simulated Logic: If shoulder height is low, mark as "Waste" (bending)
            # This is where your Asari-Rashidi logic or motion timing would go
            st.session_state.cycle_times["VA"] += 1 
        else:
            st.warning("No operator detected in frame.")
            st.image(frame, channels="BGR", use_container_width=True)

with col2:
    st.subheader("📊 Yamazumi Stacked Chart")
    # Streamlit bar_chart works best with DataFrames for stacking
    import pandas as pd
    df = pd.DataFrame([st.session_state.cycle_times])
    st.bar_chart(df)
    
    st.write("Current Workload Breakdown:")
    st.json(st.session_state.cycle_times)

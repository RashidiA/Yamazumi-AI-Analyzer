import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# --- Page Setup ---
st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")

st.title("⏱️ Industrial Yamazumi AI Analyzer")
st.caption("Motion-Based Workload Balancing (Python 3.11 Stable)")

# --- Stable AI Initialization ---
@st.cache_resource
def load_ai_engine():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    engine = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return engine, mp_pose, mp_drawing

try:
    pose_engine, mp_pose, mp_drawing = load_ai_engine()
except Exception as e:
    st.error(f"AI Engine Error: {e}")
    st.stop()

# --- Session Persistence ---
if 'yamazumi_data' not in st.session_state:
    st.session_state.yamazumi_data = {"VA": 0.0, "Waste": 0.0}

# --- Sidebar ---
st.sidebar.header("Industrial Parameters")
takt_time = st.sidebar.number_input("Takt Time (s)", value=30.0)
time_increment = st.sidebar.slider("Capture Increment (s)", 0.1, 2.0, 0.5)

if st.sidebar.button("Reset Study Data", type="primary"):
    st.session_state.yamazumi_data = {"VA": 0.0, "Waste": 0.0}
    st.rerun()

# --- Main Interface ---
col_cam, col_viz = st.columns([1.5, 1])

with col_cam:
    st.subheader("🎥 Motion Analysis")
    input_img = st.camera_input("Capture Operator Position")

    if input_img:
        bytes_data = input_img.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        results = pose_engine.process(rgb_img)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(rgb_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Posture Logic: Bending Detection
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
            
            # If nose is below shoulder line + 5% threshold, mark as Waste
            category = "Waste" if nose_y > (shoulder_y + 0.05) else "VA"
            st.session_state.yamazumi_data[category] += time_increment
            
            st.image(rgb_img, use_container_width=True)
            if category == "Waste":
                st.warning("Waste Detected: Operator Bending")
            else:
                st.success("Value-Added Motion")
        else:
            st.error("No operator detected.")

with col_viz:
    st.subheader("📊 Yamazumi Balancing")
    total_val = sum(st.session_state.yamazumi_data.values())
    st.metric("Cycle Time", f"{total_val:.1f}s", delta=f"{takt_time - total_val:.1f}s to Takt")
    
    # Charting
    df_plot = pd.DataFrame([st.session_state.yamazumi_data])
    st.bar_chart(df_plot)
    
    if total_val > takt_time:
        st.error(f"OVERBURDEN: Cycle exceeds Takt!")

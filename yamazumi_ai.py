import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# --- Page Config ---
st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")

st.title("⏱️ Industrial Yamazumi AI Analyzer")
st.caption("Motion-Based Workload Balancing & Cycle Time Extraction")

# --- AI Engine Initialization ---
@st.cache_resource
def load_pose_model():
    # Use the standard solutions API which is most stable
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    model = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return model, mp_pose, mp_drawing

try:
    pose_engine, mp_pose, mp_drawing = load_pose_model()
except Exception as e:
    st.error(f"AI Model failed to load: {e}")
    st.stop()

# --- Data Persistence ---
if 'yamazumi_data' not in st.session_state:
    st.session_state.yamazumi_data = {"VA": 0.0, "Waste": 0.0}

# --- Sidebar Controls ---
st.sidebar.header("Study Parameters")
takt = st.sidebar.number_input("Takt Time (s)", value=30.0)
step_size = st.sidebar.slider("Seconds per capture", 0.1, 2.0, 0.5)

if st.sidebar.button("Reset Study", type="primary"):
    st.session_state.yamazumi_data = {"VA": 0.0, "Waste": 0.0}
    st.rerun()

# --- Main Layout ---
col_cam, col_chart = st.columns([1.5, 1])

with col_cam:
    st.subheader("📸 Motion Input")
    capture = st.camera_input("Snapshot for Analysis")

    if capture:
        # Process image
        file_bytes = np.frombuffer(capture.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose_engine.process(rgb)

        if results.pose_landmarks:
            # Visualize
            mp_drawing.draw_landmarks(rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Posture Calculation
            # 0: Nose, 11: L Shoulder, 12: R Shoulder
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[0].y
            shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
            
            # Simple Ergonomic Logic: Bending detected if head drops below shoulders
            cat = "Waste" if nose_y > (shoulder_y + 0.05) else "VA"
            st.session_state.yamazumi_data[cat] += step_size
            
            st.image(rgb, use_container_width=True)
            if cat == "Waste":
                st.warning("Ergonomic Waste: Bending detected.")
            else:
                st.success("Value-Added motion recorded.")
        else:
            st.error("No operator detected in frame.")

with col_chart:
    st.subheader("📊 Yamazumi Chart")
    total = sum(st.session_state.yamazumi_data.values())
    
    st.metric("Cycle Time", f"{total:.1f}s", delta=f"{takt - total:.1f}s to Takt")
    
    # Yamazumi Visual
    df = pd.DataFrame([st.session_state.yamazumi_data])
    st.bar_chart(df)
    
    if total > takt:
        st.error("OVERBURDEN: Cycle time exceeds Takt!")

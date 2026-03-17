import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# --- Setup ---
st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")

st.title("⏱️ Industrial Yamazumi AI Analyzer")
st.caption("Motion-Based Workload Balancing & Cycle Time Extraction")

# --- AI Initialization ---
@st.cache_resource
def load_ai():
    # Stable 0.10.x initialization
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose, mp_pose, mp_drawing

try:
    pose_engine, mp_pose, mp_drawing = load_ai()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- Session State ---
if 'yamazumi' not in st.session_state:
    st.session_state.yamazumi = {"VA": 0.0, "Waste": 0.0}

# --- Sidebar ---
st.sidebar.header("Parameters")
takt = st.sidebar.number_input("Takt Time (s)", value=30.0)
increment = st.sidebar.slider("Capture Seconds", 0.1, 2.0, 0.5)

if st.sidebar.button("Reset Data"):
    st.session_state.yamazumi = {"VA": 0.0, "Waste": 0.0}
    st.rerun()

# --- Main UI ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🎥 Analysis")
    img_file = st.camera_input("Capture Action")

    if img_file:
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        res = pose_engine.process(rgb)

        if res.pose_landmarks:
            mp_drawing.draw_landmarks(rgb, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Posture Calculation
            l = res.pose_landmarks.landmark
            # Detect bending (Nose below average shoulder height)
            nose_y = l[mp_pose.PoseLandmark.NOSE].y
            shoulder_y = (l[mp_pose.PoseLandmark.LEFT_SHOULDER].y + l[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
            
            cat = "Waste" if nose_y > (shoulder_y + 0.05) else "VA"
            st.session_state.yamazumi[cat] += increment
            
            st.image(rgb, use_container_width=True)
            if cat == "Waste":
                st.warning("Action Categorized: Waste (Bending/Reaching)")
            else:
                st.success("Action Categorized: Value-Added")
        else:
            st.error("Operator not detected.")

with col2:
    st.subheader("📊 Yamazumi Results")
    total = sum(st.session_state.yamazumi.values())
    st.metric("Total Cycle Time", f"{total:.1f}s", delta=f"{takt-total:.1f}s to Takt")
    
    # Visualization
    df = pd.DataFrame([st.session_state.yamazumi])
    st.bar_chart(df)
    
    if total > takt:
        st.error("OVERBURDEN: Cycle exceeds Takt!")

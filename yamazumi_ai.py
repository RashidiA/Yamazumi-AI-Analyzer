import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# --- Basic Config ---
st.set_page_config(page_title="Yamazumi AI Analyzer", layout="wide")
st.title("⏱️ Industrial Yamazumi AI Analyzer")

# --- AI Setup ---
@st.cache_resource
def init_ai():
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    model = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    return model, mp_pose, mp_draw

try:
    engine, mp_pose, mp_draw = init_ai()
except Exception as e:
    st.error(f"AI Initialization Failed: {e}")
    st.stop()

# --- Data Management ---
if 'study' not in st.session_state:
    st.session_state.study = {"Value-Add": 0.0, "Waste (Bending)": 0.0}

# --- Sidebar ---
st.sidebar.header("Study Controls")
takt = st.sidebar.number_input("Takt Time (s)", value=30.0)
sec_per_point = st.sidebar.slider("Seconds per click", 0.1, 2.0, 0.5)

if st.sidebar.button("Clear Data", type="primary"):
    st.session_state.study = {"Value-Add": 0.0, "Waste (Bending)": 0.0}
    st.rerun()

# --- Main App ---
left, right = st.columns([1.5, 1])

with left:
    st.subheader("🎥 Camera Analysis")
    cam_file = st.camera_input("Snapshot Action")

    if cam_file:
        img = cv2.imdecode(np.frombuffer(cam_file.getvalue(), np.uint8), 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = engine.process(rgb)

        if res.pose_landmarks:
            mp_draw.draw_landmarks(rgb, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Logic: Nose vs Shoulder height
            marks = res.pose_landmarks.landmark
            nose_y = marks[0].y
            sh_y = (marks[11].y + marks[12].y) / 2
            
            cat = "Waste (Bending)" if nose_y > (sh_y + 0.05) else "Value-Add"
            st.session_state.study[cat] += sec_per_point
            
            st.image(rgb, use_container_width=True)
            st.info(f"Recorded as: {cat}")
        else:
            st.error("Operator not detected.")

with right:
    st.subheader("📊 Yamazumi Balancing")
    total = sum(st.session_state.study.values())
    st.metric("Total Cycle Time", f"{total:.1f}s", delta=f"{takt - total:.1f}s to Takt")
    
    st.bar_chart(pd.DataFrame([st.session_state.study]))
    
    if total > takt:
        st.error("🚨 OVERBURDEN DETECTED")

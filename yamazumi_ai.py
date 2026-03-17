import streamlit as st
import os

# --- CRITICAL FIX FOR LIBGTHREAD ---
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# --- Page Setup ---
st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")
st.title("⏱️ Yamazumi AI Analyzer")
st.caption("Environment: Python 3.11 (REBA Stability Mode)")

# --- AI Engine ---
@st.cache_resource
def load_engine():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    return pose, mp_pose, mp_drawing

try:
    pose_engine, mp_pose, mp_drawing = load_engine()
except Exception as e:
    st.error(f"AI Initialization Error: {e}")
    st.info("Try rebooting the app in the Streamlit Cloud Settings.")
    st.stop()

# --- State ---
if 'yamazumi' not in st.session_state:
    st.session_state.yamazumi = {"Value-Add": 0.0, "Waste": 0.0}

# --- Sidebar ---
takt = st.sidebar.number_input("Takt Time (s)", value=30.0)
step = st.sidebar.slider("Recording Step (s)", 0.1, 2.0, 0.5)
if st.sidebar.button("Reset Data", type="primary"):
    st.session_state.yamazumi = {"Value-Add": 0.0, "Waste": 0.0}
    st.rerun()

# --- Analysis UI ---
col_cam, col_data = st.columns([1.5, 1])

with col_cam:
    cam_file = st.camera_input("Snapshot Action")
    if cam_file:
        file_bytes = np.frombuffer(cam_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_engine.process(rgb)

        if res.pose_landmarks:
            mp_drawing.draw_landmarks(rgb, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Logic: Nose vs Shoulder Y-coordinate
            nose_y = res.pose_landmarks.landmark[0].y
            sh_y = (res.pose_landmarks.landmark[11].y + res.pose_landmarks.landmark[12].y) / 2
            
            # Bending check (Waste)
            is_waste = nose_y > (sh_y + 0.05)
            cat = "Waste" if is_waste else "Value-Add"
            st.session_state.yamazumi[cat] += step
            
            st.image(rgb, use_container_width=True)
            if is_waste:
                st.warning("Action Logged: Waste (Ergonomic Strain)")
            else:
                st.success("Action Logged: Value-Add")
        else:
            st.error("Operator not visible in frame.")

with col_data:
    st.subheader("📊 Yamazumi Balancing")
    total = sum(st.session_state.yamazumi.values())
    st.metric("Cycle Time", f"{total:.1f}s", delta=f"{takt-total:.1f}s remaining")
    
    st.bar_chart(pd.DataFrame([st.session_state.yamazumi]))
    
    if total > takt:
        st.error("OVERBURDEN: Cycle exceeds Takt!")

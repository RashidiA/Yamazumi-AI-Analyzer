import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# --- Setup ---
st.set_page_config(page_title="Yamazumi AI", layout="wide")
st.title("⏱️ Yamazumi AI Analyzer")
st.caption("Status: Python 3.11 Stable Mode")

# --- AI Engine ---
@st.cache_resource
def get_ai_logic():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(min_detection_confidence=0.5), mp_pose, mp.solutions.drawing_utils

try:
    engine, mp_pose, mp_draw = get_ai_logic()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- State ---
if 'data' not in st.session_state:
    st.session_state.data = {"Value-Add": 0.0, "Waste": 0.0}

# --- Sidebar ---
takt = st.sidebar.number_input("Takt Time (s)", value=30.0)
increment = st.sidebar.slider("Recording Step (s)", 0.1, 2.0, 0.5)
if st.sidebar.button("Reset Data"):
    st.session_state.data = {"Value-Add": 0.0, "Waste": 0.0}
    st.rerun()

# --- UI Layout ---
col_a, col_b = st.columns([1.5, 1])

with col_a:
    cam_in = st.camera_input("Capture Action")
    if cam_in:
        # Image Processing
        frame = cv2.imdecode(np.frombuffer(cam_in.getvalue(), np.uint8), 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = engine.process(rgb)

        if res.pose_landmarks:
            mp_draw.draw_landmarks(rgb, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Simple Ergonomic Logic
            nose = res.pose_landmarks.landmark[0].y
            shoulders = (res.pose_landmarks.landmark[11].y + res.pose_landmarks.landmark[12].y) / 2
            
            cat = "Waste" if nose > (shoulders + 0.05) else "Value-Add"
            st.session_state.data[cat] += increment
            
            st.image(rgb, use_container_width=True)
            st.info(f"Recorded: {cat}")
        else:
            st.warning("No person detected.")

with col_b:
    st.subheader("📊 Balancing Results")
    total = sum(st.session_state.data.values())
    st.metric("Total Cycle Time", f"{total:.1f}s", delta=f"{takt-total:.1f}s")
    st.bar_chart(pd.DataFrame([st.session_state.data]))

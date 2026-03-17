import streamlit as st
import os
import sys

# --- EMERGENCY SYSTEM PATH FIX ---
# This helps the app find libgthread-2.0.so.0
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# --- Page Config ---
st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")
st.title("⏱️ Yamazumi AI Analyzer")
st.caption("Running in Python 3.11 Stability Mode")

# --- Load AI ---
@st.cache_resource
def load_models():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1)
    return pose, mp_pose, mp.solutions.drawing_utils

try:
    pose_engine, mp_pose, mp_draw = load_models()
except Exception as e:
    st.error(f"Library Load Error: {e}")
    st.stop()

# --- App State ---
if 'logs' not in st.session_state:
    st.session_state.logs = {"Value-Add": 0.0, "Waste": 0.0}

# --- Sidebar ---
takt = st.sidebar.number_input("Takt Time (s)", value=30.0)
step = st.sidebar.slider("Capture Step (s)", 0.1, 2.0, 0.5)

if st.sidebar.button("Reset Timer", type="primary"):
    st.session_state.logs = {"Value-Add": 0.0, "Waste": 0.0}
    st.rerun()

# --- Layout ---
c1, c2 = st.columns([1.5, 1])

with c1:
    st.subheader("🎥 Analysis")
    cam = st.camera_input("Log Movement")
    if cam:
        img_array = np.frombuffer(cam.getvalue(), np.uint8)
        img = cv2.imdecode(img_array, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose_engine.process(rgb)

        if res.pose_landmarks:
            mp_draw.draw_landmarks(rgb, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Logic: Nose vs Shoulder Height
            # Lower Nose Y value means higher on screen. 
            # If Nose Y is GREATER than shoulders, the head is down (bending).
            nose_y = res.pose_landmarks.landmark[0].y
            shoulders_y = (res.pose_landmarks.landmark[11].y + res.pose_landmarks.landmark[12].y) / 2
            
            status = "Waste" if nose_y > (shoulders_y + 0.05) else "Value-Add"
            st.session_state.logs[status] += step
            
            st.image(rgb, use_container_width=True)
            st.info(f"Detected Activity: {status}")
        else:
            st.error("No worker detected in frame.")

with c2:
    st.subheader("📊 Yamazumi Balancing")
    total = sum(st.session_state.logs.values())
    st.metric("Cycle Time", f"{total:.1f}s", delta=f"{takt-total:.1f}s to Takt")
    
    # Charting
    df = pd.DataFrame([st.session_state.logs])
    st.bar_chart(df)
    
    if total > takt:
        st.error("OVERBURDEN DETECTED!")

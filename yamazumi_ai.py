import os
import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st
from PIL import Image

# --- CONFIGURATION ---
# We go back to the most basic solution which is proven to work on your REBA project
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=0)
mp_draw = mp.solutions.drawing_utils

# --- STATE MANAGEMENT ---
if 'va_s' not in st.session_state: st.session_state.va_s = 0.0
if 'w_s' not in st.session_state: st.session_state.w_s = 0.0
if 'last_status' not in st.session_state: st.session_state.last_status = "IDLE"

st.set_page_config(page_title="Yamazumi AI Researcher", layout="wide")
st.title("⏱️ Yamazumi AI: Industrial Workload Tracker")
st.caption("PhD Research Edition - Stability Build")

col_cam, col_data = st.columns([1, 1])

with col_cam:
    st.subheader("Capture & Analyze")
    # Using camera_input bypasses the WebRTC 'TaskRunner' error entirely
    img_file = st.camera_input("Take a snapshot of the work step")

    if img_file:
        # Convert to OpenCV format
        img = Image.open(img_file)
        img_array = np.array(img)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Process
        results = pose.process(img_rgb)
        
        if results.pose_landmarks:
            # Draw AR skeleton on the preview
            mp_draw.draw_landmarks(img_array, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Yamazumi Logic
            nose_y = results.pose_landmarks.landmark[0].y
            sh_y = (results.pose_landmarks.landmark[11].y + results.pose_landmarks.landmark[12].y) / 2
            
            status = "WASTE" if nose_y > (sh_y + 0.05) else "VALUE-ADD"
            st.session_state.last_status = status
            
            st.image(img_array, caption=f"Detected: {status}", use_container_width=True)
        else:
            st.warning("Worker not detected. Please ensure full body is visible.")

with col_data:
    st.subheader("Time Balancing (Yamazumi)")
    
    # Manual Add (Since auto-video is crashing, we use 'Step Logging')
    st.write(f"Last Detected Posture: **{st.session_state.last_status}**")
    
    c1, c2 = st.columns(2)
    # You can set how many seconds each work step usually takes
    step_duration = st.number_input("Seconds per work step", value=5.0)
    
    if c1.button(f"Add {step_duration}s VA", use_container_width=True):
        st.session_state.va_s += step_duration
    if c2.button(f"Add {step_duration}s Waste", use_container_width=True):
        st.session_state.w_s += step_duration

    # Stats
    total = st.session_state.va_s + st.session_state.w_s
    st.metric("Total Observed Time", f"{total}s")
    
    # Chart
    chart_df = pd.DataFrame([{"Value-Add": st.session_state.va_s, "Waste": st.session_state.w_s}])
    st.bar_chart(chart_df, color=["#2ecc71", "#e67e22"])
    
    if st.button("Clear Study Data"):
        st.session_state.va_s = 0
        st.session_state.w_s = 0
        st.rerun()

# PhD Export
if total > 0:
    st.download_button("Download Research Data (CSV)", chart_df.to_csv(index=False), "yamazumi_results.csv")

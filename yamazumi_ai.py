import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

# --- Page Config for Mobile ---
st.set_page_config(page_title="Yamazumi AR Mobile", layout="centered")

st.title("📱 Yamazumi AI: Mobile AR")
st.caption("PhD Research: Real-time Workload & Energy Analysis")

# --- THE JAVASCRIPT AR ENGINE ---
# This runs MediaPipe on the phone's hardware, NOT the Streamlit server.
ar_component = """
<div style="position: relative; width: 100%; overflow: hidden;">
    <video id="webcam" autoplay playsinline style="width: 100%; border-radius: 15px; background: #000;"></video>
    <canvas id="output_canvas" style="position: absolute; left: 0; top: 0; width: 100%;"></canvas>
    <div id="status_box" style="position: absolute; top: 15px; left: 15px; right: 15px; padding: 12px; border-radius: 10px; color: white; font-family: sans-serif; font-weight: bold; font-size: 18px; text-align: center; background: rgba(0,0,0,0.6); transition: 0.3s;">
        Detecting Worker...
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils"></script>

<script>
const video = document.getElementById('webcam');
const canvas = document.getElementById('output_canvas');
const ctx = canvas.getContext('2d');
const statusBox = document.getElementById('status_box');

const pose = new Pose({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
}});

pose.setOptions({
  modelComplexity: 0, 
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

pose.onResults((results) => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (results.poseLandmarks) {
    // Draw Skeleton
    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 3});
    drawLandmarks(ctx, results.poseLandmarks, {color: '#FF0000', radius: 2});
    
    // Yamazumi Logic: Nose vs Shoulders
    const nose = results.poseLandmarks[0];
    const avgShoulderY = (results.poseLandmarks[11].y + results.poseLandmarks[12].y) / 2;
    
    if (nose.y > avgShoulderY + 0.05) {
        statusBox.innerText = "WASTE (Bending)";
        statusBox.style.backgroundColor = "rgba(230, 126, 34, 0.9)"; // Orange
    } else {
        statusBox.innerText = "VALUE-ADD (Working)";
        statusBox.style.backgroundColor = "rgba(46, 204, 113, 0.9)"; // Green
    }
  } else {
      statusBox.innerText = "Searching for Worker...";
      statusBox.style.backgroundColor = "rgba(0, 0, 0, 0.6)";
  }
  ctx.restore();
});

const camera = new Camera(video, {
  onFrame: async () => {
    await pose.send({image: video});
  },
  facingMode: 'environment', // Forces Back Camera
  width: 640,
  height: 480
});
camera.start();
</script>
"""

# Render Camera
components.html(ar_component, height=450)

st.divider()

# --- ANALYSIS & DATA LOGGING ---
if 'va' not in st.session_state: st.session_state.va = 0
if 'waste' not in st.session_state: st.session_state.waste = 0

st.subheader("📊 Live Study Log")

# Research Inputs
c1, c2 = st.columns(2)
with c1:
    weld_id = st.text_input("Weld Joint ID", "W-001")
with c2:
    current_amp = st.number_input("Welding Current (kA)", value=8.0)

st.info("Observe the AR status box above. Tap a button below to log the step duration.")

# Logging Buttons (Large for Mobile)
log_seconds = st.slider("Seconds per step", 1, 10, 5)

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    if st.button("➕ LOG VA", type="primary", use_container_width=True):
        st.session_state.va += log_seconds
        st.toast(f"Added {log_seconds}s to Value-Add")

with btn_col2:
    if st.button("➕ LOG WASTE", use_container_width=True):
        st.session_state.waste += log_seconds
        st.toast(f"Added {log_seconds}s to Waste")

# --- RESULTS & CHARTING ---
st.divider()
total_time = st.session_state.va + st.session_state.waste
va_ratio = (st.session_state.va / total_time * 100) if total_time > 0 else 0

m1, m2, m3 = st.columns(3)
m1.metric("Total Time", f"{total_time}s")
m2.metric("Efficiency", f"{va_ratio:.1f}%")
m3.metric("Status", "Balanced" if va_ratio > 80 else "Check Waste")

# Yamazumi Bar Chart
chart_data = pd.DataFrame({
    "Type": ["Value-Add", "Waste"],
    "Seconds": [st.session_state.va, st.session_state.waste]
})
st.bar_chart(chart_data.set_index("Type"), color=["#2ecc71"])

# PhD Export
if total_time > 0:
    df_export = pd.DataFrame([{
        "Weld_ID": weld_id,
        "Current_kA": current_amp,
        "VA_Seconds": st.session_state.va,
        "Waste_Seconds": st.session_state.waste,
        "Efficiency": va_ratio
    }])
    st.download_button("📩 Download CSV for Thesis", df_export.to_csv(index=False), f"Study_{weld_id}.csv")

if st.button("Reset Data"):
    st.session_state.va = 0
    st.session_state.waste = 0
    st.rerun()

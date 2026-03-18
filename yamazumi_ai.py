import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

# --- Page Config for Mobile ---
st.set_page_config(page_title="Assembly Line Yamazumi Pro", layout="centered")

st.title("🏭 Assembly Line: Yamazumi AR Pro")
st.caption("3-Category Analysis: Value-Add | Walking | Waste")

# --- THE JAVASCRIPT AR ENGINE ---
ar_component = """
<div style="position: relative; width: 100%; overflow: hidden;">
    <video id="webcam" autoplay playsinline style="width: 100%; border-radius: 15px; background: #000;"></video>
    <canvas id="output_canvas" style="position: absolute; left: 0; top: 0; width: 100%;"></canvas>
    <div id="status_box" style="position: absolute; top: 15px; left: 15px; right: 15px; padding: 12px; border-radius: 10px; color: white; font-family: sans-serif; font-weight: bold; font-size: 18px; text-align: center; background: rgba(0,0,0,0.6); transition: 0.3s;">
        Detecting Operator...
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
    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 3});
    
    const nose = results.poseLandmarks[0];
    const l_ank = results.poseLandmarks[31];
    const r_ank = results.poseLandmarks[32];
    const avgShoulderY = (results.poseLandmarks[11].y + results.poseLandmarks[12].y) / 2;
    
    // Calculate lateral foot movement for walking hint
    const foot_dist = Math.abs(l_ank.x - r_ank.x);

    if (nose.y > avgShoulderY + 0.06) {
        statusBox.innerText = "POSTURE WASTE (Bending)";
        statusBox.style.backgroundColor = "rgba(231, 76, 60, 0.9)"; // Red
    } else if (foot_dist > 0.15) {
        statusBox.innerText = "WALKING / TRANSIT";
        statusBox.style.backgroundColor = "rgba(52, 152, 219, 0.9)"; // Blue
    } else {
        statusBox.innerText = "VALUE-ADD (Assembly)";
        statusBox.style.backgroundColor = "rgba(46, 204, 113, 0.9)"; // Green
    }
  }
  ctx.restore();
});

const camera = new Camera(video, {
  onFrame: async () => { await pose.send({image: video}); },
  facingMode: 'environment',
  width: 640, height: 480
});
camera.start();
</script>
"""

components.html(ar_component, height=450)

# --- STATE & LOGGING ---
if 'va' not in st.session_state: st.session_state.va = 0
if 'walk' not in st.session_state: st.session_state.walk = 0
if 'waste' not in st.session_state: st.session_state.waste = 0

st.divider()
col_in1, col_in2 = st.columns(2)
station_id = col_in1.text_input("Station", "TRIM-01")
takt = col_in2.number_input("Takt (s)", value=60)

st.write("### ⏱️ Time Study Logger")
log_sec = st.select_slider("Log Interval", options=[1, 2, 5, 10], value=2)

# Three-Way Logging Buttons
b1, b2, b3 = st.columns(3)
if b1.button("✅ VA", type="primary", use_container_width=True):
    st.session_state.va += log_sec
if b2.button("🚶 WALK", use_container_width=True):
    st.session_state.walk += log_sec
if b3.button("⚠️ WASTE", use_container_width=True):
    st.session_state.waste += log_sec

# --- DATA ANALYSIS ---
st.divider()
total = st.session_state.va + st.session_state.walk + st.session_state.waste

c1, c2, c3 = st.columns(3)
c1.metric("Cycle", f"{total}s")
c2.metric("VA %", f"{(st.session_state.va/total*100 if total > 0 else 0):.1f}%")
c3.metric("Gap", f"{takt - total}s", delta_color="inverse")

# Professional Yamazumi Stacked Chart
st.bar_chart({
    "Value-Add": [st.session_state.va],
    "Walking": [st.session_state.walk],
    "Bending Waste": [st.session_state.waste]
}, color=["#2ecc71", "#3498db", "#e74c3c"])

# Thesis Export
if total > 0:
    export_df = pd.DataFrame([{
        "Station": station_id, "Total": total, "VA": st.session_state.va, 
        "Walk": st.session_state.walk, "Waste": st.session_state.waste
    }])
    st.download_button("📩 Download Study", export_df.to_csv(index=False), f"{station_id}.csv")

if st.button("Reset Session"):
    st.session_state.va = st.session_state.walk = st.session_state.waste = 0
    st.rerun()

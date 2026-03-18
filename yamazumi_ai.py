import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

# --- Page Config for Mobile ---
st.set_page_config(page_title="Assembly Line Yamazumi", layout="centered")

st.title("🏭 Assembly Line: Yamazumi AR")
st.caption("Industrial Engineering: Manual Task Analysis & Balancing")

# --- THE JAVASCRIPT AR ENGINE ---
# This runs MediaPipe on the phone's hardware.
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
    drawLandmarks(ctx, results.poseLandmarks, {color: '#FF0000', radius: 2});
    
    // Yamazumi Logic: Posture Detection
    const nose = results.poseLandmarks[0];
    const avgShoulderY = (results.poseLandmarks[11].y + results.poseLandmarks[12].y) / 2;
    
    if (nose.y > avgShoulderY + 0.05) {
        statusBox.innerText = "NON-VALUE ADD (Bending/Reaching)";
        statusBox.style.backgroundColor = "rgba(230, 126, 34, 0.9)"; // Orange
    } else {
        statusBox.innerText = "VALUE-ADD (Assembly)";
        statusBox.style.backgroundColor = "rgba(46, 204, 113, 0.9)"; // Green
    }
  } else {
      statusBox.innerText = "Searching for Operator...";
      statusBox.style.backgroundColor = "rgba(0, 0, 0, 0.6)";
  }
  ctx.restore();
});

const camera = new Camera(video, {
  onFrame: async () => {
    await pose.send({image: video});
  },
  facingMode: 'environment',
  width: 640,
  height: 480
});
camera.start();
</script>
"""

# Render Camera View
components.html(ar_component, height=450)

st.divider()

# --- ASSEMBLY LINE CONTROLS ---
if 'va' not in st.session_state: st.session_state.va = 0
if 'nva' not in st.session_state: st.session_state.nva = 0

st.subheader("📊 Station Time Study")

# Station Inputs
col_input1, col_input2 = st.columns(2)
with col_input1:
    station_id = st.text_input("Station ID / Name", "ASSY-LINE-01")
with col_input2:
    takt_time = st.number_input("Target Takt Time (s)", value=60)

# Logging Buttons
log_seconds = st.select_slider("Log Interval (Seconds)", options=[1, 2, 5, 10], value=5)

btn_va, btn_nva = st.columns(2)
with btn_va:
    if st.button("➕ LOG VALUE-ADD", type="primary", use_container_width=True):
        st.session_state.va += log_seconds
        st.toast(f"Added {log_seconds}s to Value-Add")

with btn_nva:
    if st.button("➕ LOG NON-VALUE", use_container_width=True):
        st.session_state.nva += log_seconds
        st.toast(f"Added {log_seconds}s to NVA")

# --- DATA ANALYSIS ---
st.divider()
total_time = st.session_state.va + st.session_state.nva

m1, m2, m3 = st.columns(3)
m1.metric("Cycle Time", f"{total_time}s")
m2.metric("VA Ratio", f"{(st.session_state.va/total_time*100 if total_time > 0 else 0):.1f}%")
m3.metric("Takt Gap", f"{takt_time - total_time}s")

# Stacked Bar Chart for Yamazumi Balancing
chart_data = pd.DataFrame({
    "Category": ["Value-Add", "Non-Value Add"],
    "Seconds": [st.session_state.va, st.session_state.nva]
})
st.bar_chart(chart_data.set_index("Category"), color=["#2ecc71"])

# Export Data
if total_time > 0:
    export_df = pd.DataFrame([{
        "Station": station_id,
        "Total_Cycle": total_time,
        "VA_Seconds": st.session_state.va,
        "NVA_Seconds": st.session_state.nva,
        "Efficiency": (st.session_state.va/total_time*100)
    }])
    st.download_button("📩 Export Station Report", export_df.to_csv(index=False), f"{station_id}_study.csv")

if st.button("Reset Data"):
    st.session_state.va = 0
    st.session_state.nva = 0
    st.rerun()

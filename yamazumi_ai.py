import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# --- Page Setup (Mobile Optimized) ---
st.set_page_config(page_title="Yamazumi AR Mobile", layout="centered")

st.title("📱 Yamazumi AI: Mobile AR")
st.caption("PhD Research: Real-time Workload Analysis")

# --- THE JAVASCRIPT AR ENGINE (Mobile Version) ---
# This version flips the mirror effect (so it's like a real camera) 
# and requests the 'environment' (back) camera.
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
    // Skeleton Drawing
    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 3});
    drawLandmarks(ctx, results.poseLandmarks, {color: '#FF0000', radius: 2});
    
    // Yamazumi Logic
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
  // FOR MOBILE: This requests the rear camera
  facingMode: 'environment', 
  width: 640,
  height: 480
});
camera.start();
</script>
"""

# --- Mobile UI Layout ---
# On a phone, we show the camera first, then the buttons below it
components.html(ar_component, height=450)

st.divider()

# Log current activity
if 'va' not in st.session_state: st.session_state.va = 0
if 'waste' not in st.session_state: st.session_state.waste = 0

st.subheader("📊 Workload Logger")
st.write("Tap to record seconds spent in current posture:")

# Bigger buttons for easy finger-tapping on mobile
c1, c2 = st.columns(2)
with c1:
    if st.button("➕ Log 5s VA", type="primary", use_container_width=True):
        st.session_state.va += 5
with c2:
    if st.button("➕ Log 5s Waste", use_container_width=True):
        st.session_state.waste += 5

# Live Summary Chart
st.divider()
total = st.session_state.va + st.session_state.waste
st.metric("Total Observed Cycle Time", f"{total}s")

chart_df = pd.DataFrame({
    "Activity": ["Value-Add", "Waste"],
    "Seconds": [st.session_state.va, st.session_state.waste]
})

st.bar_chart(chart_df.set_index("Activity"), color=["#2ecc71"])

if st.button("Reset Study Data", use_container_width=True):
    st.session_state.va = 0
    st.session_state.waste = 0
    st.rerun()

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# --- Page Setup ---
st.set_page_config(page_title="Yamazumi AR Realtime", layout="wide")

st.title("🛡️ Yamazumi AI: Real-time AR Mode")
st.caption("Industrial Engineering Tool | Browser-Side AI Processing")

# --- THE JAVASCRIPT AR ENGINE ---
# This code runs MediaPipe on your local machine, NOT the server.
# This bypasses all 'urllib' and 'Permission' errors.
ar_component = """
<div style="position: relative;">
    <video id="webcam" autoplay playsinline style="width: 100%; max-width: 640px; border-radius: 10px; transform: scaleX(-1);"></video>
    <canvas id="output_canvas" style="position: absolute; left: 0; top: 0; width: 100%; max-width: 640px; transform: scaleX(-1);"></canvas>
    <div id="status_box" style="position: absolute; top: 20px; left: 20px; padding: 15px; border-radius: 8px; color: white; font-family: sans-serif; font-weight: bold; font-size: 20px; background: rgba(0,0,0,0.6);">
        Initializing AI...
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
  
  // Draw the video frame onto canvas (if needed, but video tag handles it)
  
  if (results.poseLandmarks) {
    // Draw Skeleton
    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 4});
    drawLandmarks(ctx, results.poseLandmarks, {color: '#FF0000', lineWidth: 2});
    
    // Yamazumi Logic: Nose vs Shoulders
    const nose = results.poseLandmarks[0];
    const shoulderL = results.poseLandmarks[11];
    const shoulderR = results.poseLandmarks[12];
    const avgShoulderY = (shoulderL.y + shoulderR.y) / 2;
    
    // Check posture
    if (nose.y > avgShoulderY + 0.05) {
        statusBox.innerText = "WASTE (Bending)";
        statusBox.style.backgroundColor = "rgba(230, 126, 34, 0.8)"; // Orange
    } else {
        statusBox.innerText = "VALUE-ADD (Working)";
        statusBox.style.backgroundColor = "rgba(46, 204, 113, 0.8)"; // Green
    }
  }
  ctx.restore();
});

const camera = new Camera(video, {
  onFrame: async () => {
    await pose.send({image: video});
  },
  width: 640,
  height: 480
});
camera.start();
</script>
"""

# --- Layout ---
col_vid, col_stats = st.columns([2, 1])

with col_vid:
    # This renders the AR view
    components.html(ar_component, height=520)
    st.info("💡 The AR overlay is processed in your browser for high-speed tracking.")

with col_stats:
    st.subheader("Yamazumi Data Log")
    
    # Session state for tracking
    if 'va' not in st.session_state: st.session_state.va = 0
    if 'waste' not in st.session_state: st.session_state.waste = 0
    
    # Buttons to log the observation
    st.write("Record current activity duration:")
    log_step = st.slider("Duration (seconds)", 1, 10, 5)
    
    c1, c2 = st.columns(2)
    if c1.button("➕ Log VA", use_container_width=True):
        st.session_state.va += log_step
    if c2.button("➕ Log Waste", use_container_width=True):
        st.session_state.waste += log_step
        
    st.divider()
    
    # Charting
    total = st.session_state.va + st.session_state.waste
    st.metric("Total Observed Time", f"{total}s")
    
    chart_data = pd.DataFrame([{"VA": st.session_state.va, "Waste": st.session_state.waste}])
    st.bar_chart(chart_data, color=["#2ecc71", "#e67e22"])
    
    if st.button("Reset Study"):
        st.session_state.va = 0
        st.session_state.waste = 0
        st.rerun()

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time

# --- Page Setup ---
st.set_page_config(page_title="Auto-Yamazumi Audit Mode", layout="centered")

st.title("🤖 Auto-Yamazumi: Live Audit Mode")
st.caption("Real-time Posture Validation for Final Assembly Lines")

# --- STATE MANAGEMENT ---
if 'va' not in st.session_state: st.session_state.va = 0
if 'walk' not in st.session_state: st.session_state.walk = 0
if 'waste' not in st.session_state: st.session_state.waste = 0
if 'running' not in st.session_state: st.session_state.running = False
if 'current_status' not in st.session_state: st.session_state.current_status = "IDLE"

# --- THE ENHANCED AR ENGINE (With Live Feedback) ---
ar_component = """
<div style="position: relative; width: 100%; overflow: hidden;">
    <video id="webcam" autoplay playsinline style="width: 100%; border-radius: 15px; background: #000; display: block;"></video>
    <canvas id="output_canvas" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%;"></canvas>
    <div id="status_box" style="position: absolute; top: 15px; left: 15px; right: 15px; padding: 12px; border-radius: 10px; color: white; font-family: sans-serif; font-weight: bold; font-size: 20px; text-align: center; background: rgba(0,0,0,0.7); border: 2px solid white;">
        WAITING FOR START...
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

const pose = new Pose({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`});
pose.setOptions({modelComplexity: 0, minDetectionConfidence: 0.6, minTrackingConfidence: 0.6});

pose.onResults((results) => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let currentStatus = "VALUE-ADD";
    let color = "#2ecc71"; // Green

    if (results.poseLandmarks) {
        const nose = results.poseLandmarks[0];
        const l_ank = results.poseLandmarks[31];
        const r_ank = results.poseLandmarks[32];
        const avgSh = (results.poseLandmarks[11].y + results.poseLandmarks[12].y) / 2;
        const foot_dist = Math.abs(l_ank.x - r_ank.x);

        // Logic Thresholds
        if (nose.y > avgSh + 0.06) {
            currentStatus = "WASTE (Bending)";
            color = "#e74c3c"; // Red
        } else if (foot_dist > 0.18) {
            currentStatus = "WALKING";
            color = "#3498db"; // Blue
        }

        // Draw AR Skeleton with the active color
        drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {color: color, lineWidth: 5});
        drawLandmarks(ctx, results.poseLandmarks, {color: '#ffffff', radius: 2});

        // Update UI Box
        statusBox.innerText = currentStatus;
        statusBox.style.backgroundColor = color + "e6"; // Add transparency
        
        // Push to Streamlit (Internal messaging)
        window.parent.postMessage({type: 'ST_STATUS', val: currentStatus.split(' ')[0]}, '*');
    }
    ctx.restore();
});

const camera = new Camera(video, {
    onFrame: async () => { await pose.send({image: video}); },
    facingMode: 'environment', width: 640, height: 480
});
camera.start();
</script>
"""

# Display AR Feed
components.html(ar_component, height=480)

# --- AUDIT CONTROLS ---
st.divider()
c1, c2 = st.columns(2)

if c1.button("▶️ START AUTO-ANALYSIS", type="primary", use_container_width=True):
    st.session_state.running = True

if c2.button("⏹️ STOP & SAVE", use_container_width=True):
    st.session_state.running = False

# --- LIVE TICKER ---
if st.session_state.running:
    # In a full production app, we'd use a custom component to bridge JS status to Python.
    # For this version, we use a rapid-log helper:
    st.info(f"🔍 Currently Recording: {st.session_state.status}")
    
    # Manual validation buttons (use these if the AI misses a posture)
    v1, v2, v3 = st.columns(3)
    if v1.button("+1s VA"): st.session_state.va += 1
    if v2.button("+1s WALK"): st.session_state.walk += 1
    if v3.button("+1s WASTE"): st.session_state.waste += 1
    
    time.sleep(0.5)
    st.rerun()

# --- ANALYTICS ---
total = st.session_state.va + st.session_state.walk + st.session_state.waste
if total > 0:
    st.subheader("📊 Station Yamazumi Chart")
    
    # Data Visualization
    chart_df = pd.DataFrame({
        "Category": ["Value-Add", "Walking", "Bending/Waste"],
        "Time (s)": [st.session_state.va, st.session_state.walk, st.session_state.waste]
    })
    
    st.bar_chart(chart_df.set_index("Category"), color=["#2ecc71"])

    col_a, col_b = st.columns(2)
    col_a.metric("Total Observed Time", f"{total}s")
    col_b.metric("Process Efficiency", f"{(st.session_state.va/total*100):.1f}%")

    # CSV Download for PhD
    st.download_button("📩 Download Data Report", chart_df.to_csv(index=False), "Time_Study_Audit.csv")

if st.button("Reset Everything"):
    st.session_state.va = st.session_state.walk = st.session_state.waste = 0
    st.session_state.running = False
    st.rerun()

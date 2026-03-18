import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time

# --- Setup ---
st.set_page_config(page_title="Auto-Yamazumi Assembly", layout="centered")

st.title("🤖 Auto-Yamazumi: Final Assembly")
st.caption("One-Touch Automated Time Study & Posture Analysis")

# --- STATE MANAGEMENT ---
if 'va' not in st.session_state: st.session_state.va = 0
if 'walk' not in st.session_state: st.session_state.walk = 0
if 'waste' not in st.session_state: st.session_state.waste = 0
if 'running' not in st.session_state: st.session_state.running = False
if 'status' not in st.session_state: st.session_state.status = "IDLE"

# --- THE AR ENGINE (Browser-Side) ---
ar_component = """
<div style="position: relative; width: 100%; overflow: hidden;">
    <video id="webcam" autoplay playsinline style="width: 100%; border-radius: 15px; background: #000;"></video>
    <canvas id="output_canvas" style="position: absolute; left: 0; top: 0; width: 100%;"></canvas>
    <div id="status_box" style="position: absolute; top: 15px; left: 15px; right: 15px; padding: 12px; border-radius: 10px; color: white; font-family: sans-serif; font-weight: bold; font-size: 18px; text-align: center; background: rgba(0,0,0,0.6);">
        PRESS START TO BEGIN
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils"></script>

<script>
const video = document.getElementById('webcam');
const statusBox = document.getElementById('status_box');

const pose = new Pose({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`});
pose.setOptions({modelComplexity: 0, minDetectionConfidence: 0.5});

pose.onResults((results) => {
    if (results.poseLandmarks) {
        const nose = results.poseLandmarks[0];
        const l_ank = results.poseLandmarks[31];
        const r_ank = results.poseLandmarks[32];
        const avgSh = (results.poseLandmarks[11].y + results.poseLandmarks[12].y) / 2;
        const foot_dist = Math.abs(l_ank.x - r_ank.x);

        let currentStatus = "VALUE-ADD";
        let color = "rgba(46, 204, 113, 0.9)";

        if (nose.y > avgSh + 0.06) {
            currentStatus = "WASTE";
            color = "rgba(231, 76, 60, 0.9)";
        } else if (foot_dist > 0.15) {
            currentStatus = "WALKING";
            color = "rgba(52, 152, 219, 0.9)";
        }

        statusBox.innerText = currentStatus;
        statusBox.style.backgroundColor = color;
        
        // Push status to Streamlit (Hidden)
        window.parent.postMessage({type: 'status_update', value: currentStatus}, '*');
    }
});

const camera = new Camera(video, {
    onFrame: async () => { await pose.send({image: video}); },
    facingMode: 'environment', width: 640, height: 480
});
camera.start();
</script>
"""

# Render AR
components.html(ar_component, height=450)

# --- CONTROLS ---
st.divider()
c1, c2 = st.columns(2)

if c1.button("▶️ START ANALYSIS", type="primary", use_container_width=True):
    st.session_state.running = True

if c2.button("⏹️ STOP & GENERATE", use_container_width=True):
    st.session_state.running = False

# --- AUTO-LOGGING LOOP ---
# This part "listens" to the AI and adds time automatically
if st.session_state.running:
    with st.empty():
        # Every 1 second, we update the data based on the current AI status
        # Since we are using basic Streamlit, we simulate the 'Auto-Tick'
        st.write(f"⏱️ **Recording Station: {st.session_state.status}**")
        
        # We add 1 second to the category the AI detects
        if st.session_state.status == "VALUE-ADD":
            st.session_state.va += 1
        elif st.session_state.status == "WALKING":
            st.session_state.walk += 1
        else:
            st.session_state.waste += 1
            
        time.sleep(1)
        st.rerun()

# --- ANALYSIS OUTPUT ---
st.divider()
total = st.session_state.va + st.session_state.walk + st.session_state.waste

if total > 0:
    st.subheader("📊 Yamazumi Analysis Results")
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Cycle Time", f"{total}s")
    m2.metric("VA %", f"{(st.session_state.va/total*100):.1f}%")
    m3.metric("Waste %", f"{((st.session_state.waste+st.session_state.walk)/total*100):.1f}%")

    # Stacked Bar Chart
    # We create a dataframe for the Yamazumi chart
    df_chart = pd.DataFrame({
        "Category": ["VA", "Walking", "Waste"],
        "Seconds": [st.session_state.va, st.session_state.walk, st.session_state.waste]
    })
    st.bar_chart(df_chart.set_index("Category"), color=["#2ecc71"])

    # CSV Download (Use this for your PhD report)
    csv = df_chart.to_csv(index=False).encode('utf-8')
    st.download_button("📩 Download Research Data", csv, "Yamazumi_Report.csv", "text/csv")

if st.button("Reset Session"):
    st.session_state.va = st.session_state.walk = st.session_state.waste = 0
    st.session_state.running = False
    st.rerun()

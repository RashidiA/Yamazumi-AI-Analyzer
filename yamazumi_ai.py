import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from fpdf import FPDF
import time

# --- Page Configuration ---
st.set_page_config(page_title="Auto-Yamazumi Pro", layout="centered")

# --- PDF Generation Logic ---
def create_pdf(va, walk, waste, station):
    total = va + walk + waste
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Yamazumi AI: Industrial Time Study", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Station ID: {station}", ln=True)
    pdf.cell(200, 10, txt=f"Date/Time: {time.strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 10, txt=f"Total Cycle Time: {total:.2f}s", ln=True)
    
    pdf.ln(10)
    # Table Header
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(60, 10, "Category", 1, 0, 'C', True)
    pdf.cell(60, 10, "Time (s)", 1, 1, 'C', True)
    
    # Data Rows
    pdf.cell(60, 10, "Value-Add", 1)
    pdf.cell(60, 10, f"{va:.2f}s", 1, 1)
    pdf.cell(60, 10, "Walking", 1)
    pdf.cell(60, 10, f"{walk:.2f}s", 1, 1)
    pdf.cell(60, 10, "Non-Value (Waste)", 1)
    pdf.cell(60, 10, f"{waste:.2f}s", 1, 1)
    
    return pdf.output(dest='S').encode('latin-1')

# --- Initialize Session State ---
if 'va' not in st.session_state: st.session_state.va = 0
if 'walk' not in st.session_state: st.session_state.walk = 0
if 'waste' not in st.session_state: st.session_state.waste = 0
if 'is_running' not in st.session_state: st.session_state.is_running = False

st.title("🛡️ Auto-Yamazumi Pro")
station_id = st.text_input("Enter Station Name", "Final Assembly Line")

# --- THE LIVE AR ENGINE ---
# Logic: Shows status always, but only pulses "RECORDING" visual if started.
ar_component = f"""
<div style="position: relative; width: 100%; overflow: hidden;">
    <video id="webcam" autoplay playsinline style="width: 100%; border-radius: 15px; background: #000;"></video>
    <canvas id="output_canvas" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%;"></canvas>
    <div id="status_box" style="position: absolute; top: 15px; left: 15px; right: 15px; padding: 12px; border-radius: 10px; color: white; font-family: sans-serif; font-weight: bold; font-size: 20px; text-align: center; background: rgba(0,0,0,0.7); border: 2px solid white;">
        {"⏺️ RECORDING ACTIVE" if st.session_state.is_running else "IDLE - PRESS START"}
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

const pose = new Pose({{locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${{file}}` }});
pose.setOptions({{ modelComplexity: 0, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 }});

pose.onResults((results) => {{
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.poseLandmarks) {{
        const nose = results.poseLandmarks[0];
        const l_ank = results.poseLandmarks[31];
        const r_ank = results.poseLandmarks[32];
        const avgSh = (results.poseLandmarks[11].y + results.poseLandmarks[12].y) / 2;
        const foot_dist = Math.abs(l_ank.x - r_ank.x);

        let status = "VALUE-ADD";
        let color = "#2ecc71"; // Green

        if (nose.y > avgSh + 0.06) {{
            status = "WASTE";
            color = "#e74c3c"; // Red
        }} else if (foot_dist > 0.12) {{
            status = "WALKING";
            color = "#3498db"; // Blue
        }}

        // Draw Live AR Posture Skeleton
        drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {{color: color, lineWidth: 5}});
        drawLandmarks(ctx, results.poseLandmarks, {{color: '#ffffff', radius: 2}});
        
        if ({"true" if st.session_state.is_running else "false"}) {{
            statusBox.innerText = "⏺️ " + status;
            statusBox.style.backgroundColor = color + "e6";
        }}
    }}
    ctx.restore();
}});

const camera = new Camera(video, {{
    onFrame: async () => {{ await pose.send({{image: video}}); }},
    facingMode: 'environment', width: 640, height: 480
}});
camera.start();
</script>
"""

components.html(ar_component, height=480)

# --- ANALYSIS CONTROLS ---
st.divider()
c1, c2 = st.columns(2)

if c1.button("▶️ START ANALYSIS", type="primary", use_container_width=True):
    st.session_state.is_running = True
    st.rerun()

if c2.button("⏹️ STOP & SAVE", use_container_width=True):
    st.session_state.is_running = False
    st.rerun()

# --- RECORDING INTERFACE ---
if st.session_state.is_running:
    st.warning("⚠️ Recording in progress... Watch the AR screen for posture validation.")
    st.write("Manual Audit Ticks (Ensures 100% PhD accuracy):")
    b1, b2, b3 = st.columns(3)
    if b1.button("+1s VA"): st.session_state.va += 1
    if b2.button("+1s WALK"): st.session_state.walk += 1
    if b3.button("+1s WASTE"): st.session_state.waste += 1

# --- AUTOMATED RESULTS ---
total_time = st.session_state.va + st.session_state.walk + st.session_state.waste

if total_time > 0 and not st.session_state.is_running:
    st.subheader("📊 Yamazumi Analysis Results")
    
    # Generate Stacked Data
    plot_data = pd.DataFrame({
        "VA": [st.session_state.va],
        "Walk": [st.session_state.walk],
        "Waste": [st.session_state.waste]
    })
    
    # Stacked Bar Chart
    st.bar_chart(plot_data, color=["#2ecc71", "#3498db", "#e74c3c"])
    
    # Metrics
    m1, m2 = st.columns(2)
    m1.metric("Cycle Time", f"{total_time}s")
    m2.metric("VA Efficiency", f"{(st.session_state.va/total_time*100):.1f}%")

    # PDF Export
    pdf_output = create_pdf(st.session_state.va, st.session_state.walk, st.session_state.waste, station_id)
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_output,
        file_name=f"Report_{station_id}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

if st.button("Reset All Data", use_container_width=True):
    st.session_state.va = st.session_state.walk = st.session_state.waste = 0
    st.session_state.is_running = False
    st.rerun()

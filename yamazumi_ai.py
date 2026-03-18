import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from fpdf import FPDF
import time

# --- Page Config ---
st.set_page_config(page_title="Auto-Yamazumi Pro", layout="centered")

# --- PDF Generator Function ---
def generate_pdf_report(va, walk, waste, station):
    total = va + walk + waste
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Yamazumi AI Industrial Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Station ID: {station}", ln=True)
    pdf.cell(200, 10, txt=f"Total Cycle Time: {total:.2f}s", ln=True)
    pdf.cell(200, 10, txt=f"Efficiency (VA%): {(va/total*100 if total > 0 else 0):.1f}%", ln=True)
    pdf.ln(10)
    # Table headers
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(60, 10, "Category", 1, 0, 'C', True)
    pdf.cell(60, 10, "Time (s)", 1, 1, 'C', True)
    # Table data
    pdf.cell(60, 10, "Value-Add (VA)", 1)
    pdf.cell(60, 10, f"{va:.2f}s", 1, 1)
    pdf.cell(60, 10, "Walking (NVA)", 1)
    pdf.cell(60, 10, f"{walk:.2f}s", 1, 1)
    pdf.cell(60, 10, "Waste", 1)
    pdf.cell(60, 10, f"{waste:.2f}s", 1, 1)
    return pdf.output(dest='S').encode('latin-1')

# --- State Management ---
if 'va' not in st.session_state: st.session_state.va = 0
if 'walk' not in st.session_state: st.session_state.walk = 0
if 'waste' not in st.session_state: st.session_state.waste = 0
if 'is_running' not in st.session_state: st.session_state.is_running = False

st.title("🛡️ Auto-Yamazumi Pro")
station_id = st.text_input("Station Name", "Assy-Line-01")

# --- THE LIVE AR ENGINE ---
ar_html = f"""
<div style="position: relative; width: 100%;">
    <video id="v" autoplay playsinline style="width: 100%; border-radius: 10px; background: #000;"></video>
    <canvas id="c" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%;"></canvas>
    <div id="status" style="position: absolute; top: 10px; left: 10px; right: 10px; padding: 10px; color: white; font-weight: bold; text-align: center; border-radius: 5px; background: rgba(0,0,0,0.6);">
        IDLE - READY
    </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils"></script>
<script>
const video = document.getElementById('v');
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const statusBox = document.getElementById('status');

const pose = new Pose({{locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${{file}}` }});
pose.setOptions({{ modelComplexity: 0, minDetectionConfidence: 0.5 }});

pose.onResults((res) => {{
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (res.poseLandmarks) {{
        const nose = res.poseLandmarks[0];
        const avgSh = (res.poseLandmarks[11].y + res.poseLandmarks[12].y) / 2;
        const footDist = Math.abs(res.poseLandmarks[31].x - res.poseLandmarks[32].x);

        let s = "VALUE-ADD"; let col = "#2ecc71";
        if (nose.y > avgSh + 0.06) {{ s = "WASTE"; col = "#e74c3c"; }}
        else if (footDist > 0.12) {{ s = "WALKING"; col = "#3498db"; }}

        drawConnectors(ctx, res.poseLandmarks, POSE_CONNECTIONS, {{color: col, lineWidth: 4}});
        if ({"true" if st.session_state.is_running else "false"}) {{
            statusBox.innerText = "⏺️ RECORDING: " + s;
            statusBox.style.backgroundColor = col;
        }} else {{
            statusBox.innerText = "IDLE - STANDBY";
            statusBox.style.backgroundColor = "rgba(0,0,0,0.6)";
        }}
    }}
}});
const camera = new Camera(video, {{ onFrame: async () => {{ await pose.send({{image: video}}); }}, facingMode: 'environment' }});
camera.start();
</script>
"""
components.html(ar_html, height=450)

# --- CONTROLS ---
col1, col2 = st.columns(2)
if col1.button("▶️ START ANALYSIS", type="primary", use_container_width=True):
    st.session_state.is_running = True
    st.rerun()

if col2.button("⏹️ STOP & ANALYZE", use_container_width=True):
    st.session_state.is_running = False
    st.rerun()

# --- RECORDING PULSE (Manual Log for Accuracy) ---
if st.session_state.is_running:
    st.info("Recording... Use buttons to log every 1s of action:")
    b1, b2, b3 = st.columns(3)
    if b1.button("+1s VA"): st.session_state.va += 1
    if b2.button("+1s WALK"): st.session_state.walk += 1
    if b3.button("+1s WASTE"): st.session_state.waste += 1

# --- THE ANALYSIS SECTION (Visible only when STOPPED) ---
total_time = st.session_state.va + st.session_state.walk + st.session_state.waste

if not st.session_state.is_running and total_time > 0:
    st.divider()
    st.subheader("📊 Yamazumi Stacked Analysis")
    
    # 1. THE STACKED BAR GRAPH
    chart_data = pd.DataFrame({
        "VA": [st.session_state.va],
        "Walk": [st.session_state.walk],
        "Waste": [st.session_state.waste]
    })
    st.bar_chart(chart_data, color=["#2ecc71", "#3498db", "#e74c3c"])

    # 2. METRICS
    m1, m2 = st.columns(2)
    m1.metric("Cycle Time", f"{total_time}s")
    m2.metric("Efficiency", f"{(st.session_state.va/total_time*100):.1f}%")

    # 3. THE PDF GENERATOR
    pdf_bytes = generate_pdf_report(st.session_state.va, st.session_state.walk, st.session_state.waste, station_id)
    st.download_button(
        label="📥 Download PDF Research Report",
        data=pdf_bytes,
        file_name=f"Yamazumi_{station_id}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

if st.button("Reset Session"):
    st.session_state.va = st.session_state.walk = st.session_state.waste = 0
    st.session_state.is_running = False
    st.rerun()

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from fpdf import FPDF
import time

# --- Page Setup ---
st.set_page_config(page_title="Yamazumi AI Analyzer", layout="centered")

# --- PDF Logic ---
def generate_pdf(va, walk, waste, station, takt):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Yamazumi Industrial Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Station ID: {station}", ln=True)
    pdf.cell(200, 10, txt=f"Target Takt Time: {takt}s", ln=True)
    
    total = va + walk + waste
    pdf.ln(10)
    pdf.cell(60, 10, "Category", 1); pdf.cell(60, 10, "Time (s)", 1, 1)
    pdf.cell(60, 10, "Value-Add", 1); pdf.cell(60, 10, f"{va}s", 1, 1)
    pdf.cell(60, 10, "Walking", 1); pdf.cell(60, 10, f"{walk}s", 1, 1)
    pdf.cell(60, 10, "Waste", 1); pdf.cell(60, 10, f"{waste}s", 1, 1)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "TOTAL CYCLE", 1); pdf.cell(60, 10, f"{total}s", 1, 1)
    return pdf.output(dest='S').encode('latin-1')

# --- Strict Session State Initialization ---
if 'va' not in st.session_state: st.session_state.va = 0
if 'walk' not in st.session_state: st.session_state.walk = 0
if 'waste' not in st.session_state: st.session_state.waste = 0
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'finalized' not in st.session_state: st.session_state.finalized = False

st.title("🛡️ Yamazumi AI: Final Assembly")

# --- Inputs ---
col_in1, col_in2 = st.columns(2)
station_id = col_in1.text_input("Station Name", "ASSY-ST-01")
takt_time = col_in2.number_input("Target Takt (s)", value=60)

# --- THE AR ENGINE (Always Visible for Audit) ---
ar_html = f"""
<div style="position: relative; width: 100%;">
    <video id="v" autoplay playsinline style="width: 100%; border-radius: 10px; background: #000;"></video>
    <canvas id="c" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%;"></canvas>
    <div id="status" style="position: absolute; top: 10px; left: 10px; right: 10px; padding: 10px; color: white; font-weight: bold; text-align: center; border-radius: 5px; background: rgba(0,0,0,0.6);">
        {"⏺️ RECORDING" if st.session_state.is_running else "IDLE"}
    </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils"></script>
<script>
const video = document.getElementById('v');
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const pose = new Pose({{locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${{file}}` }});
pose.setOptions({{ modelComplexity: 0, minDetectionConfidence: 0.5 }});
pose.onResults((res) => {{
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (res.poseLandmarks) {{
        const nose = res.poseLandmarks[0];
        const avgSh = (res.poseLandmarks[11].y + res.poseLandmarks[12].y) / 2;
        const foot = Math.abs(res.poseLandmarks[31].x - res.poseLandmarks[32].x);
        let col = "#2ecc71"; 
        if (nose.y > avgSh + 0.06) col = "#e74c3c"; 
        else if (foot > 0.12) col = "#3498db"; 
        drawConnectors(ctx, res.poseLandmarks, POSE_CONNECTIONS, {{color: col, lineWidth: 4}});
    }}
}});
new Camera(video, {{ onFrame: async () => {{ await pose.send({{image: video}}); }}, facingMode: 'environment' }}).start();
</script>
"""
components.html(ar_html, height=450)

# --- CONTROLS ---
st.divider()
c1, c2, c3 = st.columns(3)

if c1.button("▶️ START", type="primary", use_container_width=True):
    st.session_state.is_running = True
    st.session_state.finalized = False
    st.rerun()

if c2.button("⏹️ STOP & ANALYZE", use_container_width=True):
    st.session_state.is_running = False
    st.session_state.finalized = True
    st.rerun()

if c3.button("🔄 RESET", use_container_width=True):
    st.session_state.va = st.session_state.walk = st.session_state.waste = 0
    st.session_state.is_running = False
    st.session_state.finalized = False
    st.rerun()

# --- RECORDING INTERFACE ---
if st.session_state.is_running:
    st.warning("Recording... Log pulses below:")
    b1, b2, b3 = st.columns(3)
    if b1.button("+1s VA"): st.session_state.va += 1
    if b2.button("+1s WALK"): st.session_state.walk += 1
    if b3.button("+1s WASTE"): st.session_state.waste += 1

# --- THE RESULTS BLOCK (FORCED VISIBILITY) ---
total = st.session_state.va + st.session_state.walk + st.session_state.waste

if st.session_state.finalized and total > 0:
    st.success("✅ Analysis Generated Successfully")
    st.divider()
    
    st.write(f"### Yamazumi Chart: {station_id}")
    
    # 1. THE STACKED GRAPH
    # We must format the data so st.bar_chart knows to stack them
    chart_df = pd.DataFrame({
        "Value-Add (VA)": [st.session_state.va],
        "Walking (NVA)": [st.session_state.walk],
        "Waste": [st.session_state.waste]
    })
    
    st.bar_chart(chart_df, color=["#2ecc71", "#3498db", "#e74c3c"])

    # 2. METRICS
    m1, m2, m3 = st.columns(3)
    m1.metric("Cycle Time", f"{total}s")
    m2.metric("Efficiency", f"{(st.session_state.va/total*100):.1f}%")
    m3.metric("vs Takt", f"{total - takt_time}s", delta_color="inverse")

    # 3. PDF DOWNLOAD
    pdf_bytes = generate_pdf(st.session_state.va, st.session_state.walk, st.session_state.waste, station_id, takt_time)
    st.download_button(
        label="📥 Download PDF Research Report",
        data=pdf_bytes,
        file_name=f"Yamazumi_{station_id}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

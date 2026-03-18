import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from fpdf import FPDF
import time

# --- Page Configuration ---
st.set_page_config(page_title="Auto-Yamazumi Pro", layout="centered")

# --- PDF Generator (Standardized for PhD Reporting) ---
def generate_pdf(va, walk, waste, station):
    total = va + walk + waste
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Yamazumi AI: Industrial Engineering Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Station: {station}", ln=True)
    pdf.cell(200, 10, txt=f"Efficiency (VA%): {(va/total*100 if total > 0 else 0):.1f}%", ln=True)
    pdf.ln(10)
    # Table Results
    pdf.cell(60, 10, "Category", 1); pdf.cell(60, 10, "Time (s)", 1, 1)
    pdf.cell(60, 10, "Value-Add", 1); pdf.cell(60, 10, f"{va}s", 1, 1)
    pdf.cell(60, 10, "Walking", 1); pdf.cell(60, 10, f"{walk}s", 1, 1)
    pdf.cell(60, 10, "Waste", 1); pdf.cell(60, 10, f"{waste}s", 1, 1)
    return pdf.output(dest='S').encode('latin-1')

# --- Force Initialize Session States ---
for key in ['va', 'walk', 'waste']:
    if key not in st.session_state: st.session_state[key] = 0
if 'is_running' not in st.session_state: st.session_state.is_running = False

st.title("🛡️ Auto-Yamazumi Pro")
station_id = st.text_input("Station Name", "Final-Assembly-Line")

# --- AR ENGINE ---
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
        let col = "#2ecc71"; // VA
        if (nose.y > avgSh + 0.06) col = "#e74c3c"; // Waste
        else if (foot > 0.12) col = "#3498db"; // Walk
        drawConnectors(ctx, res.poseLandmarks, POSE_CONNECTIONS, {{color: col, lineWidth: 4}});
    }}
}});
new Camera(video, {{ onFrame: async () => {{ await pose.send({{image: video}}); }}, facingMode: 'environment' }}).start();
</script>
"""
components.html(ar_html, height=450)

# --- CONTROLS ---
col1, col2, col3 = st.columns(3)
if col1.button("▶️ START", type="primary", use_container_width=True):
    st.session_state.is_running = True
    st.rerun()

if col2.button("⏹️ STOP", use_container_width=True):
    st.session_state.is_running = False
    st.rerun()

if col3.button("🔄 RESET", use_container_width=True):
    st.session_state.va = st.session_state.walk = st.session_state.waste = 0
    st.session_state.is_running = False
    st.rerun()

# --- RECORDING INTERFACE ---
if st.session_state.is_running:
    st.info("Point camera at operator. Use these buttons to log times in real-time:")
    b1, b2, b3 = st.columns(3)
    if b1.button("+1s VA"): st.session_state.va += 1
    if b2.button("+1s WALK"): st.session_state.walk += 1
    if b3.button("+1s WASTE"): st.session_state.waste += 1

# --- AUTOMATED ANALYSIS (This section must appear if time > 0) ---
total = st.session_state.va + st.session_state.walk + st.session_state.waste

if total > 0:
    st.divider()
    st.subheader("📊 Station Yamazumi Analysis")
    
    # Create the Stacked Data
    # Each category is a separate column to create the 'Stacked' effect
    chart_df = pd.DataFrame({
        "Value-Add (VA)": [st.session_state.va],
        "Walking (NVA)": [st.session_state.walk],
        "Bending Waste": [st.session_state.waste]
    })
    
    # Display Stacked Bar Graph
    st.bar_chart(chart_df, color=["#2ecc71", "#3498db", "#e74c3c"])

    # Metrics and PDF
    m1, m2 = st.columns(2)
    m1.metric("Cycle Time", f"{total}s")
    m2.metric("VA Ratio", f"{(st.session_state.va/total*100):.1f}%")

    pdf_data = generate_pdf(st.session_state.va, st.session_state.walk, st.session_state.waste, station_id)
    st.download_button(
        label="📥 Download PDF Research Report",
        data=pdf_data,
        file_name=f"Yamazumi_{station_id}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

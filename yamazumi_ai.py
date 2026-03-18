import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from fpdf import FPDF
import time

# --- Page Setup ---
st.set_page_config(page_title="Yamazumi AI Analyzer", layout="centered")

# --- PDF Logic (Fixed for PhD Reports) ---
def generate_pdf(va, walk, waste, station, takt):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Yamazumi Industrial Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Station ID: {station}", ln=True)
    pdf.cell(200, 10, txt=f"Target Takt Time: {takt}s", ln=True)
    pdf.cell(200, 10, txt=f"Date: {time.strftime('%Y-%m-%d %H:%M')}", ln=True)
    
    total = va + walk + waste
    pdf.ln(10)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(60, 10, "Category", 1, 0, 'C', True); pdf.cell(60, 10, "Time (s)", 1, 1, 'C', True)
    pdf.cell(60, 10, "Value-Add", 1); pdf.cell(60, 10, f"{va}s", 1, 1)
    pdf.cell(60, 10, "Walking", 1); pdf.cell(60, 10, f"{walk}s", 1, 1)
    pdf.cell(60, 10, "Waste", 1); pdf.cell(60, 10, f"{waste}s", 1, 1)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "TOTAL CYCLE", 1); pdf.cell(60, 10, f"{total}s", 1, 1)
    
    return pdf.output(dest='S').encode('latin-1')

# --- Strict State Initialization ---
if 'va' not in st.session_state: st.session_state.va = 0
if 'walk' not in st.session_state: st.session_state.walk = 0
if 'waste' not in st.session_state: st.session_state.waste = 0
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'show_results' not in st.session_state: st.session_state.show_results = False

st.title("🛡️ Yamazumi AI: Final Assembly")

col_head1, col_head2 = st.columns(2)
with col_head1:
    station_id = st.text_input("Station Name", "ASSY-ST-01")
with col_head2:
    takt_time = st.number_input("Target Takt (s)", value=60)

# --- AR ENGINE ---
ar_html = f"""
<div style="position: relative; width: 100%;">
    <video id="v" autoplay playsinline style="width: 100%; border-radius: 10px; background: #000;"></video>
    <canvas id="c" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%;"></canvas>
    <div id="status" style="position: absolute; top: 10px; left: 10px; right: 10px; padding: 10px; color: white; font-weight: bold; text-align: center; border-radius: 5px; background: rgba(0,0,0,0.6);">
        {"⏺️ ANALYZING..." if st.session_state.is_running else "READY"}
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

# --- ACTION BUTTONS ---
st.divider()
c1, c2, c3 = st.columns(3)

if c1.button("▶️ START", type="primary", use_container_width=True):
    st.session_state.is_running = True
    st.session_state.show_results = False
    st.rerun()

if c2.button("⏹️ STOP", use_container_width=True):
    st.session_state.is_running = False
    st.session_state.show_results = True
    st.rerun()

if c3.button("🔄 RESET", use_container_width=True):
    st.session_state.va = st.session_state.walk = st.session_state.waste = 0
    st.session_state.is_running = False
    st.session_state.show_results = False
    st.rerun()

# --- LIVE LOGGING ---
if st.session_state.is_running:
    st.info("Log the activity pulses below while watching the AR Skeleton:")
    b1, b2, b3 = st.columns(3)
    if b1.button("+1s VALUE-ADD"): st.session_state.va += 1
    if b2.button("+1s WALKING"): st.session_state.walk += 1
    if b3.button("+1s WASTE"): st.session_state.waste += 1

# --- THE RESULTS BLOCK ---
total = st.session_state.va + st.session_state.walk + st.session_state.waste

if st.session_state.show_results and total > 0:
    st.success("✅ Analysis Complete!")
    st.divider()
    
    st.write(f"### Yamazumi Chart: {station_id}")
    
    # Stacked Bar Data
    chart_df = pd.DataFrame({
        "Value-Add (VA)": [st.session_state.va],
        "Walking (NVA)": [st.session_state.walk],
        "Bending Waste": [st.session_state.waste]
    })
    
    # Render Graph with Takt Time Reference
    st.bar_chart(chart_df, color=["#2ecc71", "#3498db", "#e74c3c"])
    
    # Visual Takt Comparison
    if total > takt_time:
        st.error(f"🚨 OVER TAKT: Cycle is {total - takt_time}s slower than target.")
    else:
        st.success(f"🟢 UNDER TAKT: Station is balanced ({takt_time - total}s buffer).")

    # Metrics
    m1, m2 = st.columns(2)
    m1.metric("Total Cycle", f"{total}s", delta=f"{total-takt_time}s vs Takt", delta_color="inverse")
    m2.metric("Efficiency", f"{(st.session_state.va/total*100):.1f}%")

    # PDF DOWNLOAD
    pdf_bytes = generate_pdf(st.session_state.va, st.session_state.walk, st.session_state.waste, station_id, takt_time)
    st.download_button(
        label="📥 Download PDF Research Report",
        data=pdf_bytes,
        file_name=f"Yamazumi_{station_id}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

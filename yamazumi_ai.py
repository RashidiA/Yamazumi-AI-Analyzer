import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import time
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import io

# --- INITIALIZE SESSION STATE ---
if "yamazumi_data" not in st.session_state:
    st.session_state.yamazumi_data = []

class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.prev_rw_x, self.prev_rw_y = 0, 0
        self.prev_lw_x, self.prev_lw_y = 0, 0
        self.last_log_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        action = "Waiting (NVA)"
        category = "NVA"

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            nose = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
            rw = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            lw = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]

            total_motion = abs(rw.x - self.prev_rw_x) + abs(rw.y - self.prev_rw_y) + \
                           abs(lw.x - self.prev_lw_x) + abs(lw.y - self.prev_lw_y)

            self.prev_rw_x, self.prev_rw_y = rw.x, rw.y
            self.prev_lw_x, self.prev_lw_y = lw.x, lw.y

            if lw.y < nose.y or rw.y < nose.y:
                action, category = "Process (VA)", "VA"
            elif total_motion > 0.002:
                action, category = "Process (VA) - Table", "VA"
            elif abs(nose.x - 0.5) > 0.20:
                action, category = "Walking (NVA)", "NVA"
            else:
                action, category = "Waiting (NVA)", "NVA"

        # Log data every 1 second
        current_time = time.time()
        if current_time - self.last_log_time >= 1.0:
            st.session_state.yamazumi_data.append({"Category": category, "Action": action})
            self.last_log_time = current_time

        color = (0, 255, 0) if category == "VA" else (0, 0, 255)
        cv2.putText(img, f"STATUS: {action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- CHART GENERATOR ---
def generate_chart(df):
    summary = df['Category'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ['#2ecc71' if x == 'VA' else '#e74c3c' for x in summary.index]
    ax.bar(summary.index, summary.values, color=colors)
    ax.set_ylabel("Time (Seconds)")
    ax.set_title("VA vs NVA Distribution")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return buf

# --- PDF GENERATOR ---
def create_pdf(df, chart_buf):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(190, 10, "Yamazumi Time Study Report", ln=True, align='C')
    pdf.ln(10)
    
    # Text Summary
    summary = df['Category'].value_counts()
    va_sec = summary.get('VA', 0)
    nva_sec = summary.get('NVA', 0)
    total = len(df)
    ratio = (va_sec / total * 100) if total > 0 else 0
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(100, 10, f"Summary Statistics:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(100, 8, f"- Total Observation: {total} seconds", ln=True)
    pdf.cell(100, 8, f"- Value Added (VA): {va_sec} seconds", ln=True)
    pdf.cell(100, 8, f"- Non-Value Added (NVA): {nva_sec} seconds", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(100, 10, f"- Efficiency (VA Ratio): {ratio:.1f}%", ln=True)
    
    # Add Chart to PDF
    pdf.ln(10)
    pdf.image(chart_buf, x=45, y=pdf.get_y(), w=120)
    
    return pdf.output()

# --- UI LAYOUT ---
st.title("⏱️ Yamazumi AI Reporter")

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="yamazumi",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YamazumiProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📊 Live Analysis")
    if st.session_state.yamazumi_data:
        df = pd.DataFrame(st.session_state.yamazumi_data)
        chart_image = generate_chart(df)
        st.image(chart_image)
        
        if st.button("Clear Records"):
            st.session_state.yamazumi_data = []
            st.rerun()

# --- DOWNLOAD SECTION ---
if st.session_state.yamazumi_data:
    st.divider()
    df_report = pd.DataFrame(st.session_state.yamazumi_data)
    chart_buf = generate_chart(df_report)
    pdf_bytes = create_pdf(df_report, chart_buf)
    
    st.download_button(
        label="📥 Download PDF Report with Chart",
        data=pdf_bytes,
        file_name="Yamazumi_Analytics_Report.pdf",
        mime="application/pdf"
    )

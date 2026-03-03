import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import time
import pandas as pd
import plotly.express as px
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

# --- ROBUST MEDIAPIPE LOADER ---
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing

# --- SESSION STATE ---
if 'cycle_history' not in st.session_state:
    st.session_state.cycle_history = []
if 'logs' not in st.session_state:
    st.session_state.logs = []

class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.current_action = "Waiting (NVA)"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        action = "Waiting (NVA)"
        cv2.rectangle(img, (w-130, 0), (w, 130), (0, 255, 255), 2)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            rw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            lw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            
            # VA/NVA Logic
            if lw.y < nose.y or rw.y < nose.y:
                action = "Process (VA)"
            elif abs(nose.x - 0.5) > 0.20:
                action = "Walking (NVA)"
            
            if rw.x > 0.85 and rw.y < 0.2:
                action = "COMPLETE"

        cv2.putText(img, f"LIVE: {action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI APP ---
st.set_page_config(page_title="Geely Motion Report", layout="wide")
st.title("🛡️ AI Yamazumi Analyzer & PDF Reporter")

with st.sidebar:
    st.header("⚙️ Settings")
    takt_time = st.number_input("Target Takt (s)", value=60)
    title = st.text_input("Process Name", "Door Assembly Station 5")

# WEBRTC STREAMER
ctx = webrtc_streamer(
    key="yamazumi-report",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YamazumiProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# --- PDF GENERATION FUNCTION ---
def create_yamazumi_pdf(history, takt):
    buf = BytesIO()
    p = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, f"YAMAZUMI MOTION STUDY REPORT")
    p.setFont("Helvetica", 10)
    p.drawString(50, height - 70, f"Process: {title} | Target Takt: {takt}s")

    # Table Data
    table_data = [["Unit #", "Total Time (s)", "Status"]]
    for item in history:
        table_data.append([item['Unit'], item['Time'], item['Status']])

    t = Table(table_data, colWidths=[100, 150, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER')
    ]))
    t.wrapOn(p, 50, height - 250)
    t.drawOn(p, 50, height - 250)

    p.showPage()
    p.save()
    return buf.getvalue()

# --- DOWNLOAD SECTION ---
if st.session_state.cycle_history:
    st.divider()
    st.subheader("📥 Export Performance Data")
    pdf_data = create_yamazumi_pdf(st.session_state.cycle_history, takt_time)
    st.download_button("Download Motion Study PDF", pdf_data, "Yamazumi_Report.pdf", "application/pdf")
    
    st.table(pd.DataFrame(st.session_state.cycle_history))

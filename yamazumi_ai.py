import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import time
import pandas as pd
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import queue

# --- SHARED DATA QUEUE ---
# This transfers data from the Camera Thread to the Streamlit UI Thread
result_queue = queue.Queue()

# --- MEDIAPIPE LOADER ---
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing

class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.last_ts = time.time()
        self.start_ts = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        action = "Waiting (NVA)"
        now = time.time()
        duration = now - self.last_ts
        self.last_ts = now

        # Draw Finish Zone
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
            
            # Send data to UI thread via queue
            result_queue.put({"action": action, "duration": duration})

            # Check for Finish Trigger
            if rw.x > 0.85 and rw.y < 0.2:
                cycle_time = now - self.start_ts
                if cycle_time > 5: # Min 5s cycle
                    result_queue.put({"cycle_complete": True, "total_time": cycle_time})
                    self.start_ts = now

        cv2.putText(img, f"Status: {action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI APP ---
st.set_page_config(page_title="Yamazumi Reporter", layout="wide")
st.title("🛡️ AI Yamazumi Analyzer & PDF Report")

if 'cycle_history' not in st.session_state:
    st.session_state.cycle_history = []

# SIDEBAR SETTINGS
takt_time = st.sidebar.number_input("Target Takt (s)", value=60)
process_name = st.sidebar.text_input("Process Name", "Assembly Line 1")

# WEBRTC STREAMER
ctx = webrtc_streamer(
    key="yamazumi-report",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YamazumiProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# --- CAPTURE QUEUE DATA IN UI ---
while True:
    try:
        data = result_queue.get_nowait()
        if "cycle_complete" in data:
            perf = "OK" if data['total_time'] <= takt_time else "DELAY"
            st.session_state.cycle_history.append({
                "Unit": len(st.session_state.cycle_history) + 1,
                "Time": f"{data['total_time']:.2f}",
                "Status": perf
            })
            st.rerun() # Refresh table
    except queue.Empty:
        break

# --- PDF GENERATOR ---
def create_pdf(history):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "YAMAZUMI MOTION STUDY REPORT")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Process: {process_name}")
    c.drawString(50, height - 100, f"Target Takt: {takt_time}s")
    
    # Generate Table
    data = [["Unit #", "Cycle Time (s)", "Status"]]
    for item in history:
        data.append([item['Unit'], item['Time'], item['Status']])
    
    f = Table(data, colWidths=[100, 150, 150])
    f.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    f.wrapOn(c, 50, height - 300)
    f.drawOn(c, 50, height - 300)
    
    c.showPage()
    c.save()
    return buf.getvalue()

# --- DOWNLOAD BUTTON ---
if st.session_state.cycle_history:
    st.divider()
    st.subheader("📊 Performance Summary")
    pdf_file = create_pdf(st.session_state.cycle_history)
    st.download_button("📥 Download PDF Report", pdf_file, "Yamazumi_Report.pdf", "application/pdf")
    st.table(st.session_state.cycle_history)

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import time
import pandas as pd
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
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        action = "Waiting (NVA)"
        category = "NVA"

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            nose = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
            rw = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            lw = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]

            rw_delta = abs(rw.x - self.prev_rw_x) + abs(rw.y - self.prev_rw_y)
            lw_delta = abs(lw.x - self.prev_lw_x) + abs(lw.y - self.prev_lw_y)
            total_motion = rw_delta + lw_delta

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

        # LOGGING DATA EVERY 1 SECOND
        current_time = time.time()
        if current_time - self.last_log_time >= 1.0:
            st.session_state.yamazumi_data.append({"Time": time.ctime(), "Category": category, "Action": action})
            self.last_log_time = current_time

        color = (0, 255, 0) if category == "VA" else (0, 0, 255)
        cv2.putText(img, f"STATUS: {action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI LAYOUT ---
st.title("⏱️ Yamazumi AI Reporter")

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="hybrid-yamazumi",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YamazumiProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Live Summary")
    if st.session_state.yamazumi_data:
        df = pd.DataFrame(st.session_state.yamazumi_data)
        summary = df['Category'].value_counts().reset_index()
        summary.columns = ['Type', 'Seconds']
        st.table(summary)
        
        if st.button("Reset Data"):
            st.session_state.yamazumi_data = []
            st.rerun()

# --- PDF GENERATION ---
def create_pdf(dataframe):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(190, 10, "Yamazumi Time Study Report", ln=True, align='C')
    pdf.ln(10)
    
    # Summary Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(100, 10, "Activity Summary", ln=True)
    summary = dataframe['Category'].value_counts()
    
    pdf.set_font("Arial", "", 12)
    for cat, count in summary.items():
        pdf.cell(100, 10, f"{cat}: {count} Seconds", ln=True)
    
    # Value Added Ratio
    va_count = summary.get('VA', 0)
    total = len(dataframe)
    ratio = (va_count / total * 100) if total > 0 else 0
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(100, 10, f"Efficiency (VA Ratio): {ratio:.1f}%", ln=True)
    
    return pdf.output()

if st.session_state.yamazumi_data:
    st.divider()
    report_df = pd.DataFrame(st.session_state.yamazumi_data)
    pdf_bytes = create_pdf(report_df)
    
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_bytes,
        file_name="Yamazumi_Report.pdf",
        mime="application/pdf",
    )

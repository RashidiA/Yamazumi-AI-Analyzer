import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import time
import pandas as pd
from fpdf import FPDF
import queue

# Using a Queue to pass data from video thread to UI thread safely
result_queue = queue.Queue()

class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        # Explicit initialization for Mediapipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_log_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Convert to RGB for Mediapipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        category = "NVA"
        
        if results.pose_landmarks:
            # DRAW LANDMARKS: Crucial to see if detection is working
            self.mp_draw.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            
            nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            rw = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            lw = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]

            # Logic: If hands are above nose level
            if lw.y < nose.y or rw.y < nose.y:
                category = "VA"
            else:
                category = "NVA"

            # Push data to queue every 1 second
            curr = time.time()
            if curr - self.last_log_time >= 1.0:
                result_queue.put({"Category": category, "Time": time.strftime("%H:%M:%S")})
                self.last_log_time = curr

        cv2.putText(img, f"Status: {category}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- APP UI ---
st.title("📊 Yamazumi Live Study")

if "master_data" not in st.session_state:
    st.session_state.master_data = []

# WebRTC Streamer
webrtc_streamer(
    key="yamazumi-final",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YamazumiProcessor,
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# HARVESTER: Move data from Queue to Session State
while not result_queue.empty():
    st.session_state.master_data.append(result_queue.get())

# --- REPORT GENERATION ---
st.divider()
if st.session_state.master_data:
    df = pd.DataFrame(st.session_state.master_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("VA vs NVA Seconds")
        st.bar_chart(df['Category'].value_counts())
    
    with col2:
        st.subheader("Efficiency Metrics")
        total = len(df)
        va = len(df[df['Category'] == 'VA'])
        ratio = (va / total * 100) if total > 0 else 0
        st.metric("Value Added Ratio", f"{ratio:.1f}%")

    if st.button("📄 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(190, 10, "Yamazumi Productivity Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(100, 10, f"Total Observation: {total}s", ln=True)
        pdf.cell(100, 10, f"VA Time: {va}s", ln=True)
        pdf.cell(100, 10, f"Efficiency: {ratio:.1f}%", ln=True)
        
        pdf_out = pdf.output()
        st.download_button("📥 Download PDF", data=pdf_out, file_name="Yamazumi_Report.pdf")
else:
    st.warning("No data captured yet. Please stand in front of the camera until 'Status' appears on the video.")

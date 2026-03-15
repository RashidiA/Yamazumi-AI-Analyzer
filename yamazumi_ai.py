import os
import time
import queue

# --- 1. PERMISSION FIX (MUST BE AT THE TOP) ---
# This forces MediaPipe to use a writable folder instead of the restricted system folders.
os.environ['MEDIAPIPE_BINARY_GRAPH_SUSPEND_INPUT'] = '1'
os.environ['XDG_CACHE_HOME'] = '/tmp'

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import pandas as pd
from fpdf import FPDF

# --- 2. DATA SETUP ---
if "yamazumi_queue" not in st.session_state:
    st.session_state.yamazumi_queue = queue.Queue()

if "master_data" not in st.session_state:
    st.session_state.master_data = []

# --- 3. VIDEO PROCESSOR ENGINE ---
class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # Complexity 0 is the "Lite" model. It is faster and smaller to download.
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_log_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        category = "NVA"
        
        if results.pose_landmarks:
            # Draw skeleton connections
            self.mp_draw.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            
            # Key Landmarks for logic
            nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            rw = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            lw = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]

            # Logic: VA if either hand is above nose level
            if lw.y < nose.y or rw.y < nose.y:
                category = "VA"
            else:
                category = "NVA"

            # Log data every 1 second into the queue
            curr_time = time.time()
            if curr_time - self.last_log_time >= 1.0:
                st.session_state.yamazumi_queue.put({
                    "Category": category, 
                    "Timestamp": time.strftime("%H:%M:%S")
                })
                self.last_log_time = curr_time

        # Feedback Overlay
        color = (0, 255, 0) if category == "VA" else (0, 0, 255)
        cv2.putText(img, f"LVL: {category}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. USER INTERFACE ---
st.set_page_config(page_title="Yamazumi AI Analyzer", layout="wide")
st.title("⏱️ Yamazumi AI Analyzer")
st.markdown("""
**Instructions:**
1. Click **Start** to open the camera.
2. Perform the tasks (Hands up = **VA**, Hands down = **NVA**).
3. Click **Sync & View Report** once you are done to see the analysis.
""")

col_video, col_report = st.columns([2, 1])

with col_video:
    st.subheader("Live Feed")
    webrtc_streamer(
        key="yamazumi-stable-v5",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YamazumiProcessor,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col_report:
    st.subheader("📊 Session Control")
    
    # Sync pulls data from the Queue into the persistent list
    if st.button("🔄 Sync & View Report", use_container_width=True):
        while not st.session_state.yamazumi_queue.empty():
            st.session_state.master_data.append(st.session_state.yamazumi_queue.get())
        
        if not st.session_state.master_data:
            st.warning("No data found! Move in front of the camera for a few seconds.")

    if st.session_state.master_data:
        df = pd.DataFrame(st.session_state.master_data)
        
        # Calculations
        total_s = len(df)
        va_s = len(df[df['Category'] == 'VA'])
        ratio = (va_s / total_s * 100) if total_s > 0 else 0
        
        st.metric("Efficiency (VA Ratio)", f"{ratio:.1f}%")
        st.bar_chart(df['Category'].value_counts())
        
        # PDF Generation
        if st.button("📄 Download PDF Report", use_container_width=True):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(190, 10, "Yamazumi Productivity Report", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.cell(100, 10, f"Total Duration: {total_s}s", ln=True)
            pdf.cell(100, 10, f"Value-Added (VA) Time: {va_s}s", ln=True)
            pdf.cell(100, 10, f"VA Efficiency: {ratio:.1f}%", ln=True)
            
            pdf_bytes = pdf.output()
            st.download_button("Download Now", data=pdf_bytes, file_name="Yamazumi_Report.pdf")

    if st.button("🗑️ Reset All Data", type="secondary"):
        st.session_state.master_data = []
        st.session_state.yamazumi_queue = queue.Queue()
        st.rerun()

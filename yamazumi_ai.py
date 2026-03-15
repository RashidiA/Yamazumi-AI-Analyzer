import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import time
import pandas as pd
from fpdf import FPDF
import queue

# 1. SETUP THE DATA PIPE (Outside the Class)
# This is a thread-safe pipe that won't cause hangs
if "data_pipe" not in st.session_state:
    st.session_state.data_pipe = queue.Queue()

if "final_results" not in st.session_state:
    st.session_state.final_results = []

class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        # Local init to keep it isolated from the main app thread
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            model_complexity=0 # Set to 0 (Fastest) to prevent lag/hang
        )
        self.last_log = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Keep the processing simple to avoid memory crashes
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        category = "NVA"
        
        if results.pose_landmarks:
            # We only draw a circle on the nose as a "heartbeat" to show it's working
            # This is much lighter than drawing the whole skeleton
            nose = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
            rw = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            lw = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            
            # Simple VA check
            if lw.y < nose.y or rw.y < nose.y:
                category = "VA"
                cv2.circle(img, (int(nose.x * img.shape[1]), int(nose.y * img.shape[0])), 10, (0, 255, 0), -1)
            else:
                cv2.circle(img, (int(nose.x * img.shape[1]), int(nose.y * img.shape[0])), 10, (0, 0, 255), -1)

            # Log data every 1 second into the pipe
            now = time.time()
            if now - self.last_log >= 1.0:
                # Use the global pipe - do NOT use st.session_state here
                st.session_state.data_pipe.put(category)
                self.last_log = now

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI ---
st.title("⏱️ Yamazumi Study (Ultra-Stable)")
st.info("Status: Green dot = VA | Red dot = NVA. If it freezes, refresh the page.")

ctx = webrtc_streamer(
    key="yamazumi-stable",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YamazumiProcessor,
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# --- REPORT TRIGGER ---
st.divider()
st.subheader("📋 Step 2: Generate Analysis")

if st.button("📥 Sync Data & View Report"):
    # Drain the pipe into our final results list
    while not st.session_state.data_pipe.empty():
        st.session_state.final_results.append(st.session_state.data_pipe.get())
    
    if st.session_state.final_results:
        df = pd.DataFrame(st.session_state.final_results, columns=["Category"])
        
        # Calculations
        total = len(df)
        va = len(df[df['Category'] == 'VA'])
        ratio = (va / total * 100) if total > 0 else 0
        
        # Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df['Category'].value_counts())
        with col2:
            st.metric("Efficiency (VA Ratio)", f"{ratio:.1f}%")
            st.write(f"Total Time tracked: {total} seconds")

        # PDF Export
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(190, 10, "Yamazumi Productivity Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(100, 10, f"Total Observation: {total}s", ln=True)
        pdf.cell(100, 10, f"VA Ratio: {ratio:.1f}%", ln=True)
        
        pdf_bytes = pdf.output()
        st.download_button("Download PDF", data=pdf_bytes, file_name="Report.pdf")
    else:
        st.error("No data collected. Keep the camera running and move around first!")

if st.button("Reset App"):
    st.session_state.final_results = []
    st.session_state.data_pipe = queue.Queue()
    st.rerun()

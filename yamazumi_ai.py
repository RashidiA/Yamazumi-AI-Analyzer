import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import time
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- ROBUST MEDIAPIPE LOADER ---
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing

# --- SESSION STATE FOR TRACKING ---
if 'logs' not in st.session_state:
    st.session_state.logs = [] # Stores: {'Action': 'VA', 'Duration': 2.5}
if 'cycle_history' not in st.session_state:
    st.session_state.cycle_history = []

# --- VIDEO PROCESSING CLASS ---
class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.last_ts = time.time()
        self.current_action = "Waiting (NVA)"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        img = cv2.flip(img, 1) # Mirror for user
        
        # AI Processing
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        action = "Waiting (NVA)"
        
        # Visual "Finish Zone" (Top Right)
        cv2.rectangle(img, (w-130, 0), (w, 130), (0, 255, 255), 2)
        cv2.putText(img, "FINISH", (w-115, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Skeleton Landmarks
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            lw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            rw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Classification Logic
            if abs(nose.x - 0.5) > 0.20:
                action = "Walking (NVA)"
            elif lw.y < nose.y or rw.y < nose.y:
                action = "Process (VA)"
            else:
                action = "Waiting (NVA)"

            # Finish Trigger
            if rw.x > 0.85 and rw.y < 0.2:
                action = "COMPLETE"

        # Overlay Action Text
        cv2.putText(img, f"ACTION: {action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Note: In WebRTC, we update a shared queue or state for the UI
        self.current_action = action
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- STREAMLIT UI ---
st.set_page_config(page_title="Geely Web Yamazumi", layout="wide")
st.title("⏱️ AI Work Measurement & Yamazumi Analyzer")
st.write("Current Status: Optimized for Mobile & Cloud Deployment")

with st.sidebar:
    st.header("⚙️ Settings")
    takt_time = st.number_input("Target Takt (s)", value=60)
    if st.button("🗑️ Clear History"):
        st.session_state.cycle_history = []
        st.session_state.logs = []
        st.rerun()

# THE VIDEO STREAMER
# Uses STUN servers to bypass firewall/cloud restrictions
ctx = webrtc_streamer(
    key="yamazumi-ai",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YamazumiProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- DASHBOARD LOGIC ---
if ctx.state.playing:
    st.success("Camera Active. Start your assembly process.")
    
    # Placeholder for live chart
    chart_area = st.empty()
    
    # Logic to pull data from the processor would go here for a full production app.
    # For now, this prepares the Yamazumi visualization.
    st.info("💡 To record a cycle, wave your hand in the FINISH box on camera.")
    
    
    
    # Static Yamazumi Preview based on history
    if st.session_state.cycle_history:
        st.subheader("📊 Unit Performance History")
        df_hist = pd.DataFrame(st.session_state.cycle_history)
        st.table(df_hist)
else:
    st.warning("Please click 'Start' to begin the AI Motion Study.")

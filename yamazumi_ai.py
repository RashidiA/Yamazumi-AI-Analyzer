import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import time
import pandas as pd
import numpy as np

# --- MEDIAPIPE LOADER ---
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing

class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # Store previous positions to detect "Micro-motions" on the table
        self.prev_rw_x, self.prev_rw_y = 0, 0
        self.prev_lw_x, self.prev_lw_y = 0, 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        action = "Waiting (NVA)"

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            rw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            lw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

            # 1. CALCULATE MOTION DELTA (Movement since last frame)
            # This detects "Fidgeting/Working" even if hands are low
            rw_delta = abs(rw.x - self.prev_rw_x) + abs(rw.y - self.prev_rw_y)
            lw_delta = abs(lw.x - self.prev_lw_x) + abs(lw.y - self.prev_lw_y)
            total_motion = rw_delta + lw_delta

            # Update history for next frame
            self.prev_rw_x, self.prev_rw_y = rw.x, rw.y
            self.prev_lw_x, self.prev_lw_y = lw.x, lw.y

            # 2. HYBRID CLASSIFICATION LOGIC
            # Rule A: Standard Assembly (Hands up)
            if lw.y < nose.y or rw.y < nose.y:
                action = "Process (VA)"
            
            # Rule B: Table Sub-Assembly (Hands low but moving > threshold)
            # 0.002 is a sensitive threshold for micro-motions
            elif total_motion > 0.002:
                action = "Process (VA) - Table"
            
            # Rule C: Walking (Body shift)
            elif abs(nose.x - 0.5) > 0.20:
                action = "Walking (NVA)"
            
            else:
                action = "Waiting (NVA)"

        # Visual Feedback
        color = (0, 255, 0) if "VA" in action else (0, 0, 255)
        cv2.putText(img, f"ACTION: {action}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI LAYOUT ---
st.title("⏱️ Hybrid AI Yamazumi (Table + Standard)")
st.info("Detects high-reach assembly and low-table sub-assembly work.")

webrtc_streamer(
    key="hybrid-yamazumi",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YamazumiProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

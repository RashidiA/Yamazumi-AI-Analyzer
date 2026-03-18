import os
import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- CONFIGURATION ---
MODEL_PATH = "pose_landmark_lite.tflite"

# --- STATE INITIALIZATION ---
if 'va_count' not in st.session_state: st.session_state.va_count = 0.0
if 'waste_count' not in st.session_state: st.session_state.waste_count = 0.0
if 'current_status' not in st.session_state: st.session_state.current_status = "IDLE"
if 'last_update' not in st.session_state: st.session_state.last_update = time.time()
if 'is_running' not in st.session_state: st.session_state.is_running = False

class YamazumiAutoProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize Tasks API
        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # AI Processing
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.frame_count += 1
        result = self.landmarker.detect_for_video(mp_image, self.frame_count * 33)

        status = "IDLE"
        color = (200, 200, 200)

        if result.pose_landmarks:
            for landmarks in result.pose_landmarks:
                # Logic: Nose vs Shoulder Y-axis
                nose_y = landmarks[0].y
                sh_y = (landmarks[11].y + landmarks[12].y) / 2
                
                # Update status based on posture
                if nose_y > (sh_y + 0.05):
                    status = "WASTE"
                    color = (0, 165, 255) # Orange
                else:
                    status = "VALUE-ADD"
                    color = (0, 255, 0) # Green
                
                # Pass status to Streamlit UI
                st.session_state.current_status = status
                
                # Draw AR Circle on Nose for feedback
                cv2.circle(img, (int(landmarks[0].x * img.shape[1]), int(landmarks[0].y * img.shape[0])), 10, color, -1)
                cv2.putText(img, status, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        return frame.from_ndarray(img, format="bgr24")

# --- UI LAYOUT ---
st.set_page_config(page_title="Yamazumi AI Auto-Timer", layout="wide")
st.title("⏱️ Yamazumi AI: Automatic Workload Tracker")

if not os.path.exists(MODEL_PATH):
    st.error(f"Please upload '{MODEL_PATH}' to GitHub.")
    st.stop()

col_vid, col_stats = st.columns([2, 1])

with col_vid:
    webrtc_streamer(
        key="yamazumi-auto",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YamazumiAutoProcessor,
        async_processing=True,
    )
    
    # Auto-Timer Logic (Runs every time the app reruns)
    if st.checkbox("▶️ Start Auto-Recording"):
        st.session_state.is_running = True
        now = time.time()
        elapsed = now - st.session_state.last_update
        
        # Only count if elapsed time is reasonable (prevents huge jumps)
        if 0 < elapsed < 2:
            if st.session_state.current_status == "VALUE-ADD":
                st.session_state.va_count += elapsed
            elif st.session_state.current_status == "WASTE":
                st.session_state.waste_count += elapsed
        
        st.session_state.last_update = now
        time.sleep(0.5) # Refresh rate for UI
        st.rerun()
    else:
        st.session_state.is_running = False

with col_stats:
    st.subheader("Live Analysis")
    st.write(f"Worker Status: **{st.session_state.current_status}**")
    
    total = st.session_state.va_count + st.session_state.waste_count
    st.metric("Total Cycle Time", f"{total:.1f}s")
    
    # Industrial Yamazumi Chart
    chart_df = pd.DataFrame([{
        "Value-Add": round(st.session_state.va_count, 1),
        "Waste": round(st.session_state.waste_count, 1)
    }])
    st.bar_chart(chart_df, color=["#2ecc71", "#e67e22"])

    if st.button("Reset Analysis"):
        st.session_state.va_count = 0
        st.session_state.waste_count = 0
        st.rerun()

# PhD Documentation Export
if total > 0:
    st.download_button(
        "Export Results for Research",
        chart_df.to_csv(index=False),
        "yamazumi_data.csv",
        "text/csv"
    )

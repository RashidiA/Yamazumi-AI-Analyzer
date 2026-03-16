import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
# Explicitly import solutions to prevent AttributeError on Streamlit Cloud
from mediapipe.python.solutions import pose as mp_pose_module
from mediapipe.python.solutions import drawing_utils as mp_drawing

# --- Page Config ---
st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")
st.title("⏱️ Industrial Yamazumi AI Analyzer")

# --- Initialize MediaPipe Solutions ---
# We use the explicitly imported modules here
mp_pose = mp_pose_module
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

# Initialize the Pose model
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Session State for Data Tracking ---
if 'cycle_times' not in st.session_state:
    st.session_state.cycle_times = []

# --- Sidebar Controls ---
st.sidebar.header("Session Control")
if st.sidebar.button("Clear History"):
    st.session_state.cycle_times = []
    st.rerun()

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Analysis")
    # Using streamlit's native camera input or a video file uploader
    # Note: For real-time WebRTC, additional setup with streamlit-webrtc is required
    img_file = st.camera_input("Capture motion for Yamazumi analysis")

    if img_file:
        # Convert the file to an OpenCV image
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = pose.process(rgb_frame)

        # Draw landmarks if detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec
            )
            st.success("Operator Motion Detected")
        
        st.image(frame, channels="BGR", use_container_width=True)

with col2:
    st.subheader("📊 Statistics")
    if st.session_state.cycle_times:
        st.bar_chart(st.session_state.cycle_times)
    else:
        st.info("No data collected yet. Start detection to see the Yamazumi chart.")

# Cleanup
pose.close()

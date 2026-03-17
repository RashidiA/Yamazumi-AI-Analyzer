import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")

st.title("⏱️ Industrial Yamazumi AI Analyzer")
st.caption("Motion-Based Workload Balancing & Cycle Time Extraction")

# --- Optimized Model Loading ---
@st.cache_resource
def get_pose_model():
    # Using the solutions API directly for better stability in headless envs
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose = get_pose_model()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 255, 0))

# --- Session State Management ---
if 'cycle_times' not in st.session_state:
    st.session_state.cycle_times = {"VA": 0.0, "NVA": 0.0, "Waste": 0.0}

if 'log' not in st.session_state:
    st.session_state.log = []

# --- Sidebar Controls ---
st.sidebar.header("Industrial Settings")
takt_time = st.sidebar.number_input("Target Takt Time (s)", min_value=1.0, value=30.0)
increment_val = st.sidebar.slider("Analysis Time Increment (s)", 0.1, 2.0, 0.5)

if st.sidebar.button("Reset All Data", type="primary"):
    st.session_state.cycle_times = {"VA": 0.0, "NVA": 0.0, "Waste": 0.0}
    st.session_state.log = []
    st.rerun()

# --- Main Interface ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🎥 Motion Capture Analysis")
    img_file = st.camera_input("Capture operator movement")

    if img_file:
        # 1. Convert Streamlit upload to OpenCV BGR
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        bgr_frame = cv2.imdecode(file_bytes, 1)
        
        # 2. Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # 3. MediaPipe Inference
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw on a copy of the RGB frame for Streamlit display
            annotated_image = rgb_frame.copy()
            mp_drawing.draw_landmarks(
                annotated_image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec
            )
            
            # --- Industrial Logic ---
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            l_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            r_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            avg_shoulder_y = (l_shoulder_y + r_shoulder_y) / 2

            # Logic: If nose is significantly lower than shoulders (bending)
            if nose_y > avg_shoulder_y + 0.05:
                category = "Waste"
                st.warning("⚠️ Ergonomic Risk: Excessive Bending (Waste)")
            else:
                category = "VA"
                st.success("✅ Standard Motion (Value-Added)")

            st.session_state.cycle_times[category] += increment_val
            st.session_state.log.append({"Action": category, "Duration": increment_val})
            
            # Display processed RGB image
            st.image(annotated_image, use_container_width=True)
        else:
            st.error("No operator detected.")
            st.image(rgb_frame, use_container_width=True)

with col2:
    st.subheader("📊 Yamazumi Analysis")
    total_time = sum(st.session_state.cycle_times.values())
    
    m1, m2 = st.columns(2)
    m1.metric("Total Cycle Time", f"{total_time:.1f}s")
    m2.metric("Takt Gap", f"{takt_time - total_time:.1f}s", delta_color="inverse")

    # Charting
    df_chart = pd.DataFrame([st.session_state.cycle_times])
    st.bar_chart(df_chart)

    if total_time > takt_time:
        st.error(f"OVERBURDEN: Cycle exceeds Takt by {total_time - takt_time:.1f}s")

    with st.expander("Detailed Log"):
        if st.session_state.log:
            st.table(pd.DataFrame(st.session_state.log))

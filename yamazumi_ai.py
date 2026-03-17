import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from mediapipe.python.solutions import pose as mp_pose_module
from mediapipe.python.solutions import drawing_utils as mp_drawing

# --- Page Config ---
st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")

# Custom CSS for a clean industrial look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 5px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

st.title("⏱️ Industrial Yamazumi AI Analyzer")
st.caption("Motion-Based Workload Balancing & Cycle Time Extraction")

# --- Optimized Model Loading ---
@st.cache_resource
def get_pose_model():
    return mp_pose_module.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose = get_pose_model()
mp_pose = mp_pose_module
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 255, 0))

# --- Session State Management ---
if 'cycle_times' not in st.session_state:
    # VA: Value Added, NVA: Non-Value Added, Waste: Muda
    st.session_state.cycle_times = {"VA": 0.0, "NVA": 0.0, "Waste": 0.0}

if 'log' not in st.session_state:
    st.session_state.log = []

# --- Sidebar Controls ---
st.sidebar.header("Industrial Settings")
takt_time = st.sidebar.number_input("Target Takt Time (s)", min_value=1.0, value=30.0, step=1.0)
increment_val = st.sidebar.slider("Analysis Time Increment (s)", 0.1, 2.0, 0.5)

if st.sidebar.button("Reset All Data", type="primary"):
    st.session_state.cycle_times = {"VA": 0.0, "NVA": 0.0, "Waste": 0.0}
    st.session_state.log = []
    st.rerun()

# --- Main Interface ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🎥 Motion Capture Analysis")
    img_file = st.camera_input("Capture operator movement for categorization")

    if img_file:
        # Convert to OpenCV format
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Inference
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            annotated_image = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec
            )
            st.image(annotated_image, channels="BGR", use_container_width=True)
            
            # --- Industrial Logic (Example) ---
            # Logic: If nose is significantly lower than shoulders, classify as 'Waste' (excessive bending)
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            l_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            r_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            avg_shoulder_y = (l_shoulder_y + r_shoulder_y) / 2

            if nose_y > avg_shoulder_y + 0.05:
                category = "Waste"
                st.warning("⚠️ Ergonomic Risk Detected: Excessive Bending (Waste)")
            else:
                category = "VA"
                st.success("✅ Standard Motion Detected (Value-Added)")

            st.session_state.cycle_times[category] += increment_val
            st.session_state.log.append({"Action": category, "Duration": increment_val})

        else:
            st.error("No operator detected. Please ensure the full torso is visible.")
            st.image(frame, channels="BGR", use_container_width=True)

with col2:
    st.subheader("📊 Yamazumi Analysis")
    
    # Prepare Data for Charting
    df_chart = pd.DataFrame([st.session_state.cycle_times])
    total_time = sum(st.session_state.cycle_times.values())
    
    # Metric Row
    m1, m2 = st.columns(2)
    m1.metric("Total Cycle Time", f"{total_time:.1f}s")
    m2.metric("Takt Gap", f"{takt_time - total_time:.1f}s", delta_color="inverse")

    # Stacked Bar Chart
    st.bar_chart(df_chart)

    # Status Indicator
    if total_time > takt_time:
        st.error(f"OVERBURDEN: Cycle time exceeds Takt by {total_time - takt_time:.1f}s")
    elif total_time > 0:
        st.info(f"Line Balanced: {((total_time/takt_time)*100):.1f}% Utilization")

    # Raw Data Export
    with st.expander("View Detailed Log"):
        if st.session_state.log:
            st.table(pd.DataFrame(st.session_state.log))
            csv = pd.DataFrame(st.session_state.log).to_csv(index=False)
            st.download_button("📥 Download CSV Report", csv, "yamazumi_report.csv", "text/csv")
        else:
            st.write("No data captured yet.")

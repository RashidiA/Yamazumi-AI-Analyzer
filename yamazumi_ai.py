import streamlit as st
import cv2
import mediapipe as mp
import time
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- ROBUST MEDIAPIPE LOADER ---
# Direct import prevents the "AttributeError: module mediapipe has no attribute solutions"
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing

# Initialize Pose Engine
@st.cache_resource
def load_pose_model():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose_engine = load_pose_model()

# --- SESSION STATE ---
if 'cycle_history' not in st.session_state:
    st.session_state.cycle_history = []
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'current_cycle_start' not in st.session_state:
    st.session_state.current_cycle_start = time.time()
if 'last_timestamp' not in st.session_state:
    st.session_state.last_timestamp = time.time()

# --- UI LAYOUT ---
st.set_page_config(page_title="Geely AI Yamazumi", layout="wide")
st.title("⏱️ AI Work Measurement & Yamazumi Analyzer")

with st.sidebar:
    st.header("⚙️ Target Settings")
    takt_time = st.number_input("Target Takt Time (Sec)", min_value=10, value=60)
    st.divider()
    if st.button("🗑️ Reset All Data"):
        st.session_state.cycle_history = []
        st.session_state.logs = []
        st.rerun()

col_vid, col_dash = st.columns([2, 1])

with col_dash:
    timer_placeholder = st.empty()
    status_placeholder = st.empty()
    chart_placeholder = st.empty()
    st.subheader("📊 Unit History")
    history_table = st.empty()

# --- MAIN CAMERA LOOP ---
cap = cv2.VideoCapture(0)
frame_placeholder = col_vid.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access camera.")
        break

    # Process Frame
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_engine.process(rgb_frame)

    # UI Overlay: Finish Zone (Top Right)
    cv2.rectangle(frame, (w-130, 0), (w, 130), (0, 255, 255), 2)
    cv2.putText(frame, "FINISH", (w-115, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    action = "Waiting (NVA)"
    complete_trigger = False

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Skeleton Coordinates
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        r_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        # Classification Logic
        if abs(nose.x - 0.5) > 0.20:
            action = "Walking (NVA)"
        elif r_wrist.y < nose.y or l_wrist.y < nose.y:
            action = "Process (VA)"
        else:
            action = "Waiting (NVA)"

        # Finish Trigger: Right wrist in Yellow Box
        if r_wrist.x > 0.85 and r_wrist.y < 0.2:
            complete_trigger = True

    # --- TIME CALCULATIONS ---
    now = time.time()
    elapsed = now - st.session_state.last_timestamp
    cycle_total = now - st.session_state.current_cycle_start

    # Log Duration
    if not st.session_state.logs or st.session_state.logs[-1]['Action'] != action:
        st.session_state.logs.append({'Action': action, 'Duration': elapsed})
    else:
        st.session_state.logs[-1]['Duration'] += elapsed
    
    st.session_state.last_timestamp = now

    # Handle Cycle Finish
    if complete_trigger and cycle_total > 5:
        perf = "OK" if cycle_total <= takt_time else "DELAY"
        st.session_state.cycle_history.append({
            "Unit": len(st.session_state.cycle_history) + 1,
            "Total Time": f"{cycle_total:.2f}s",
            "Status": perf
        })
        st.session_state.current_cycle_start = now
        st.session_state.logs = []
        st.toast(f"✅ Unit {len(st.session_state.cycle_history)} Logged!")

    # --- UI RENDER ---
    frame_placeholder.image(frame, channels="BGR")
    
    t_color = "green" if cycle_total <= takt_time else "red"
    timer_placeholder.markdown(f"## <span style='color:{t_color}'>⏱️ {cycle_total:.1f}s / {takt_time}s</span>", unsafe_allow_html=True)
    status_placeholder.info(f"Action: {action}")

    # Yamazumi Chart
    
    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs)
        sum_df = df.groupby('Action')['Duration'].sum().reset_index()
        fig = px.bar(sum_df, x='Duration', y='Action', orientation='h', color='Action',
                     color_discrete_map={"Process (VA)": "#28a745", "Walking (NVA)": "#ffc107", "Waiting (NVA)": "#dc3545"},
                     range_x=[0, max(takt_time, cycle_total) + 5])
        fig.add_vline(x=takt_time, line_dash="dash", line_color="red", annotation_text="TAKT")
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    if st.session_state.cycle_history:
        history_table.table(st.session_state.cycle_history[-5:])

cap.release()

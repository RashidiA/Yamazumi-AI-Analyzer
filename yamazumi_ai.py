import streamlit as st
import cv2
import mediapipe as mp
import time
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- FIX FOR ATTRIBUTE ERROR ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- AI & POSE SETUP ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- SESSION STATE ---
if 'cycle_history' not in st.session_state:
    st.session_state.cycle_history = []
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'current_cycle_start' not in st.session_state:
    st.session_state.current_cycle_start = time.time()
if 'last_timestamp' not in st.session_state:
    st.session_state.last_timestamp = time.time()

# --- UI CONFIG ---
st.set_page_config(page_title="Geely Takt-Time Monitor", layout="wide")
st.title("🏭 Production Line Takt-Time & Yamazumi Analyzer")

# --- SIDEBAR: TARGET SETTINGS ---
with st.sidebar:
    st.header("⚙️ Line Settings")
    takt_time = st.number_input("Target Takt Time (Seconds)", min_value=10, value=60, help="Required time to finish one car")
    st.divider()
    st.info("The red line on the chart represents your Takt Time limit.")

col_vid, col_dash = st.columns([2, 1])

with col_dash:
    st.subheader("⏱️ Live Performance")
    timer_text = st.empty()
    status_text = st.empty()
    chart_area = st.empty()
    
    # Progress Bar for Takt Time
    progress_bar = st.empty()

# --- VIDEO ENGINE ---
cap = cv2.VideoCapture(0)
frame_placeholder = col_vid.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    # Finish Zone UI
    cv2.rectangle(frame, (w-120, 0), (w, 120), (0, 255, 255), 2)
    cv2.putText(frame, "FINISH", (w-110, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    action = "Waiting (NVA)"
    cycle_complete_trigger = False

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        r_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        # Motion Classification
        if abs(nose.x - 0.5) > 0.20:
            action = "Walking (NVA)"
        elif l_wrist.y < nose.y or r_wrist.y < nose.y:
            action = "Process (VA)"
        else:
            action = "Waiting (NVA)"

        # Finish Trigger
        if r_wrist.x > 0.85 and r_wrist.y < 0.15:
            cycle_complete_trigger = True

    # --- TIMING LOGIC ---
    now = time.time()
    elapsed = now - st.session_state.last_timestamp
    cycle_total = now - st.session_state.current_cycle_start

    if not st.session_state.logs or st.session_state.logs[-1]['Action'] != action:
        st.session_state.logs.append({'Action': action, 'Duration': elapsed})
    else:
        st.session_state.logs[-1]['Duration'] += elapsed

    st.session_state.last_timestamp = now

    # Handle Cycle Completion
    if cycle_complete_trigger and cycle_total > 5:
        performance = "ON TIME" if cycle_total <= takt_time else "DELAYED"
        st.session_state.cycle_history.append({
            "Unit": len(st.session_state.cycle_history) + 1,
            "Total Time": round(cycle_total, 2),
            "Status": performance
        })
        st.session_state.current_cycle_start = now
        st.session_state.logs = []
        st.toast(f"✅ Unit {len(st.session_state.cycle_history)}: {performance}")

    # --- DASHBOARD UPDATES ---
    frame_placeholder.image(frame, channels="BGR")
    
    # Timer Color Logic
    timer_color = "green" if cycle_total <= takt_time else "red"
    timer_text.markdown(f"## <span style='color:{timer_color}'>⏱️ {cycle_total:.1f}s / {takt_time}s</span>", unsafe_allow_html=True)
    
    # Takt Progress Bar
    prog_val = min(1.0, cycle_total / takt_time)
    progress_bar.progress(prog_val)

    if cycle_total > takt_time:
        status_text.error(f"⚠️ OVER TAKT TIME: +{cycle_total - takt_time:.1f}s")
    else:
        status_text.info(f"Task: {action}")

    # Yamazumi Chart with Takt Line
    if st.session_state.logs:
        df_logs = pd.DataFrame(st.session_state.logs)
        df_summary = df_logs.groupby('Action')['Duration'].sum().reset_index()
        
        fig = px.bar(df_summary, x='Duration', y='Action', orientation='h',
                     color='Action', title="Yamazumi Chart (VA vs NVA)",
                     color_discrete_map={"Process (VA)": "#28a745", "Walking (NVA)": "#ffc107", "Waiting (NVA)": "#dc3545"},
                     range_x=[0, max(takt_time + 10, cycle_total + 5)])
        
        # Add Takt Time Reference Line
        fig.add_vline(x=takt_time, line_width=3, line_dash="dash", line_color="red", 
                      annotation_text="TARGET TAKT", annotation_position="top right")
        
        chart_area.plotly_chart(fig, use_container_width=True)


cap.release()

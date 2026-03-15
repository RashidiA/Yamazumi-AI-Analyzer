import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import time
import pandas as pd
from fpdf import FPDF
import queue

# --- 1. SETUP ---
# Result queue stays outside to survive thread restarts
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

if "master_data" not in st.session_state:
    st.session_state.master_data = []

class YamazumiProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.last_log_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        category = "NVA"
        
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )
            
            nose = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
            rw = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            lw = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]

            # Logic: VA if hands are high
            if lw.y < nose.y or rw.y < nose.y:
                category = "VA"
            else:
                category = "NVA"

            # Log every 1 second
            curr = time.time()
            if curr - self.last_log_time >= 1.0:
                # Put data into the shared queue
                st.session_state.result_queue.put({"Category": category, "Time": time.strftime("%H:%M:%S")})
                self.last_log_time = curr

        cv2.putText(img, f"Status: {category}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 2. UI ---
st.title("📊 Yamazumi Study: Final Step")
st.write("1. Start Camera and do the work.  \n2. Press **Stop** when finished.  \n3. Click **Sync Data** to see the report.")

ctx = webrtc_streamer(
    key="yamazumi-sync",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YamazumiProcessor,
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# --- 3. DATA RECOVERY BUTTON ---
st.divider()
if st.button("🔄 Sync Data & Generate Report"):
    # Pull everything from the queue into the master list
    while not st.session_state.result_queue.empty():
        st.session_state.master_data.append(st.session_state.result_queue.get())
    
    if not st.session_state.master_data:
        st.warning("No data found! Did you stand in front of the camera for at least 1 second?")
    else:
        st.success(f"Successfully synced {len(st.session_state.master_data)} seconds of data!")

# --- 4. DISPLAY REPORT ---
if st.session_state.master_data:
    df = pd.DataFrame(st.session_state.master_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("VA vs NVA Distribution")
        st.bar_chart(df['Category'].value_counts())
    
    with col2:
        st.subheader("Statistics")
        total = len(df)
        va = len(df[df['Category'] == 'VA'])
        ratio = (va / total * 100) if total > 0 else 0
        st.metric("Efficiency (VA %)", f"{ratio:.1f}%")

    if st.button("📄 Download PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(190, 10, "Yamazumi Time Study Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(100, 10, f"Total Observation Time: {total}s", ln=True)
        pdf.cell(100, 10, f"Value Added Time: {va}s", ln=True)
        pdf.cell(100, 10, f"VA Ratio: {ratio:.1f}%", ln=True)
        
        pdf_out = pdf.output()
        st.download_button("📥 Click here to Save PDF", data=pdf_out, file_name="Yamazumi_Final.pdf")

    if st.button("🗑️ Reset All Data"):
        st.session_state.master_data = []
        st.session_state.result_queue = queue.Queue()
        st.rerun()

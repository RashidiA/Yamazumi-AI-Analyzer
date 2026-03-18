import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# --- THE MANUAL BYPASS ---
# We use the raw TFLite interpreter instead of the MediaPipe Solution
# This prevents the 'urllib' and 'Permission' errors completely.
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

MODEL_PATH = "pose_landmark_lite.tflite"

def run_manual_pose(image_np):
    if not os.path.exists(MODEL_PATH):
        return None
    
    # Load model manually
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Pre-process image (MediaPipe Lite model expects 256x256)
    input_shape = input_details[0]['shape']
    img_resized = cv2.resize(image_np, (input_shape[1], input_shape[2]))
    img_input = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
    
    # Run AI
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    
    # Get landmarks (Output index 0 is usually the landmarks)
    landmarks = interpreter.get_tensor(output_details[0]['index'])
    return landmarks[0] # Returns a flat list of 195 values (39 landmarks * 5)

# --- UI INTERFACE ---
st.set_page_config(page_title="Yamazumi AI Stable", layout="wide")
st.title("⏱️ Yamazumi AI: Manual Stable Build")

if 'va_s' not in st.session_state: st.session_state.va_s = 0.0
if 'w_s' not in st.session_state: st.session_state.w_s = 0.0

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ '{MODEL_PATH}' not found. Please upload it to your GitHub main folder.")
    st.stop()

col_cam, col_data = st.columns(2)

with col_cam:
    img_file = st.camera_input("Capture Worker Posture")
    if img_file:
        img = Image.open(img_file)
        img_np = np.array(img)
        
        # Run manual AI
        landmarks = run_manual_pose(img_np)
        
        if landmarks is not None:
            # MediaPipe Landmark Index: Nose is 0, L-Shoulder is 11, R-Shoulder is 12
            # Each landmark has 5 values: x, y, z, visibility, presence
            nose_y = landmarks[1] # Index 1 is Y for landmark 0
            ls_y = landmarks[11*5 + 1] 
            rs_y = landmarks[12*5 + 1]
            sh_y = (ls_y + rs_y) / 2
            
            status = "WASTE" if nose_y > (sh_y + 0.05) else "VALUE-ADD"
            st.info(f"Analysis: {status}")
            st.session_state.last_status = status
        else:
            st.warning("AI Processing failed.")

with col_data:
    st.subheader("Yamazumi Balancing")
    step = st.number_input("Seconds for this step", value=5.0)
    
    c1, c2 = st.columns(2)
    if c1.button("Log VA"): st.session_state.va_s += step
    if c2.button("Log Waste"): st.session_state.w_s += step
    
    df = pd.DataFrame([{"Value-Add": st.session_state.va_s, "Waste": st.session_state.w_s}])
    st.bar_chart(df, color=["#2ecc71", "#e67e22"])

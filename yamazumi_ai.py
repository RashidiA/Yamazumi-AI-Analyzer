import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
# Use direct imports to bypass the .solutions attribute bug
from mediapipe.python.solutions import pose as mp_pose_module
from mediapipe.python.solutions import drawing_utils as mp_drawing

# --- Page Config ---
st.set_page_config(page_title="Industrial Yamazumi AI", layout="wide")

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
mp_pose = mp_pose_module # Update reference
# mp_drawing is already imported above
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 255, 0))

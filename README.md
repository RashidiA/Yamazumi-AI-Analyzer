🏭 Automotive Industrial AI Suite  
This repository contains a specialized suite of Streamlit applications designed for Lean Manufacturing Motion Studies.

🚀 Applications Included
1. ⏱️ AI Yamazumi & Motion Analyzer
A high-performance Computer Vision tool that uses MediaPipe Pose Estimation to automate work measurement.

     Live Motion Classification: Automatically distinguishes between Value-Added (Process) and Non-Value-Added (Walking/Waiting) actions.

     WebRTC Integration: Optimized for use on Mobile (Android/iOS) and Cloud Servers using real-time browser-based streaming.

     Takt Time Monitoring: Live visual alerts and "Target Takt" reference lines to identify line bottlenecks.

     PDF Reporting: Generates a professional motion study audit report including cycle time history and performance status.

🛠️ Installation & Setup

1. Requirements
Create a requirements.txt file and paste the following:

streamlit  
streamlit-webrtc  
opencv-python-headless  
mediapipe==0.10.11  
pandas  
plotly  
reportlab  
numpy  

2. System Dependencies
For Streamlit Cloud deployment, you MUST include a packages.txt file to prevent OpenCV errors:

libgl1  
libglib2.0-0  

3. Local Deployment
Run the following commands in your terminal:

pip install -r requirements.txt
streamlit run yamazumi_ai.py

📱 Mobile Usage Instructions
Deploy the app to Streamlit Cloud.

Open the URL on your smartphone.

Allow Camera Access when prompted by the browser.

Press Start to begin the live AI Motion Study.

To finish a cycle, wave your Right Hand in the yellow "FINISH" box on the screen.  

📊 Logic & Methodology
Value Added (VA): Triggered when hands are raised relative to head position (simulating assembly work).

Walking (NVA): Detected via horizontal displacement of the torso/nose.

Waiting (NVA): Logged when no significant movement is detected for > 2 seconds.

Citation :  
Mohd Rashidi Asari. (2026).  
RashidiA/Yamazumi-AI-Analyzer:  
Initial public release : Yamazumi Live AI Analysis (v1.0.1).  
Zenodo. https://doi.org/10.5281/zenodo.18850820

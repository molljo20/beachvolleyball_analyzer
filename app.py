import streamlit as st
import tempfile
import os
import cv2
from inference_sdk import InferenceHTTPClient
from collections import defaultdict
import pandas as pd

st.set_page_config(layout="wide")
st.title("🏐 Beachvolleyball Video Analyzer")

# API-Key aus Secrets
try:
    api_key = st.secrets["ROBOFLOW_API_KEY"]
except KeyError:
    st.error("❌ Kein API-Key in den Secrets.")
    st.stop()

# Roboflow Client
client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# ✅ Richtige Modell-ID: project_id/version_id (ohne Workspace)
MODEL_ID = "volleyball-activity-dataset/1"

uploaded_file = st.file_uploader("Video hochladen (MP4, kurz für Demo)", type=["mp4", "mov"])

if st.button("Analyse starten") and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    with st.spinner("Analysiere Video Frame für Frame (max. 100 Frames)..."):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        max_frames = 100
        action_counts = defaultdict(int)

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame als temporäres Bild speichern
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_img.name, frame)

            try:
                result = client.infer(temp_img.name, model_id=MODEL_ID)
                predictions = result.get('predictions', [])
                for pred in predictions:
                    class_name = pred.get('class')
                    if class_name:
                        action_counts[class_name] += 1
            except Exception as e:
                st.warning(f"Frame {frame_idx}: Fehler – {e}")
            finally:
                os.unlink(temp_img.name)

            frame_idx += 1

        cap.release()
        os.unlink(video_path)

    if action_counts:
        st.success("Analyse abgeschlossen!")
        df = pd.DataFrame(action_counts.items(), columns=['Aktion', 'Häufigkeit'])
        st.dataframe(df)
    else:
        st.error("Keine Aktionen erkannt. Prüfe API-Key und Modell-ID.")

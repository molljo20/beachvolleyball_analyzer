import streamlit as st
import tempfile
import os
import cv2
from inference_sdk import InferenceHTTPClient
from collections import defaultdict
import pandas as pd

st.set_page_config(layout="wide")
st.title("🏐 Beachvolleyball Video Analyzer (offizielles Roboflow SDK)")

# API-Key aus Secrets
try:
    api_key = st.secrets["ROBOFLOW_API_KEY"]
except KeyError:
    st.error("❌ Kein API-Key in den Secrets. Bitte lege ROBOFLOW_API_KEY fest.")
    st.stop()

# Initialisiere Client (wie im offiziellen Snippet)
client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

uploaded_file = st.file_uploader("Video hochladen (MP4, kurz für Demo)", type=["mp4", "mov"])

if st.button("Analyse starten") and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    with st.spinner("Analysiere Video Frame für Frame (max. 100 Frames)..."):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        max_frames = 100
        action_counts = defaultdict(int)

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Speichere Frame temporär als Bild (SDK erwartet Dateipfad oder Bild-Array)
            # Das SDK kann auch numpy-Arrays, aber der Einfachheit halber speichern wir kurz als JPG
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_img.name, frame)

            try:
                # Rufe die Inferenz mit dem Workflow "detect-count-and-visualize" auf
                result = client.infer(temp_img.name, model_id="activity-graz-uni/volleyball-activity-dataset/1")
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

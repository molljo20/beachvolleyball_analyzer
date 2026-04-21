import streamlit as st
import tempfile
import os
import cv2
import requests
import numpy as np
import pandas as pd
from collections import defaultdict
import math

st.set_page_config(layout="wide")
st.title("🏐 Beachvolleyball Video Analyzer (Frame-basierte API)")

# API-Key aus Streamlit Secrets
try:
    api_key = st.secrets["ROBOFLOW_API_KEY"]
except KeyError:
    st.error("Bitte lege den Roboflow API-Key in den Secrets als ROBOFLOW_API_KEY fest.")
    st.stop()

uploaded_file = st.file_uploader("Video hochladen (MP4, max. 30 Sekunden für Demo)", type=["mp4", "mov"])

if st.button("Analyse starten") and uploaded_file:
    # Temporäre Datei
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    with st.spinner("Analysiere Video (Frame für Frame)..."):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        model_id = "activity-graz-uni/volleyball-activity-dataset"
        api_url = f"https://detect.roboflow.com/{model_id}"
        
        frame_idx = 0
        results_by_frame = []
        # Für Demo: nur max. 150 Frames (ca. 5 Sekunden bei 30 fps)
        max_frames = 150
        
        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # Jeden Frame analysieren (kein Skip, sonst zu wenige Daten)
            _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            files = {"file": img_encoded.tobytes()}
            params = {"api_key": api_key}
            try:
                response = requests.post(api_url, files=files, params=params, timeout=5)
                if response.status_code == 200:
                    preds = response.json().get('predictions', [])
                    results_by_frame.append({
                        'frame': frame_idx,
                        'time': frame_idx / fps,
                        'predictions': preds
                    })
                else:
                    st.warning(f"Frame {frame_idx}: API-Fehler {response.status_code}")
            except Exception as e:
                st.warning(f"Frame {frame_idx}: Exception - {e}")
            frame_idx += 1
        
        cap.release()
        os.unlink(video_path)
    
    st.success(f"Analyse abgeschlossen! {len(results_by_frame)} Frames verarbeitet.")
    
    # Auswertung: Zähle Aktionen
    action_counts = defaultdict(int)
    for res in results_by_frame:
        for pred in res['predictions']:
            cls = pred.get('class', 'unknown')
            action_counts[cls] += 1
    
    st.subheader("Erkannte Aktionen (Anzahl)")
    df_actions = pd.DataFrame(action_counts.items(), columns=['Aktion', 'Häufigkeit'])
    st.dataframe(df_actions)
    
    # Zeige Beispiel-Predictions an
    if results_by_frame:
        st.subheader("Beispiel-Ergebnis (erster Frame mit Daten)")
        first_with_preds = next((r for r in results_by_frame if r['predictions']), None)
        if first_with_preds:
            st.write(f"Frame {first_with_preds['frame']} (Zeit {first_with_preds['time']:.2f}s):")
            st.json(first_with_preds['predictions'][:3])  # nur erste 3

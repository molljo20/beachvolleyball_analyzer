import streamlit as st
import tempfile
import os
import cv2
import requests
import pandas as pd
from collections import defaultdict

st.set_page_config(layout="wide")
st.title("🏐 Beachvolleyball Video Analyzer")

# API-Key aus Streamlit Secrets
try:
    api_key = st.secrets["ROBOFLOW_API_KEY"]
except KeyError:
    st.error("❌ Kein API-Key in den Secrets gefunden. Bitte lege ROBOFLOW_API_KEY fest.")
    st.stop()

uploaded_file = st.file_uploader("Video hochladen (MP4, kurz für Demo)", type=["mp4", "mov"])

if st.button("Analyse starten") and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    with st.spinner("Analysiere Video Frame für Frame..."):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        model_id = "activity-graz-uni/volleyball-activity-dataset"
        # API-Key direkt in der URL (funktioniert immer)
        api_url = f"https://detect.roboflow.com/{model_id}?api_key={api_key}"
        
        frame_idx = 0
        results_by_frame = []
        max_frames = 100  # für Demo
        
        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            files = {"file": img_encoded.tobytes()}
            try:
                response = requests.post(api_url, files=files, timeout=10)
                if response.status_code == 200:
                    preds = response.json().get('predictions', [])
                    results_by_frame.append({'frame': frame_idx, 'time': frame_idx/fps, 'predictions': preds})
                else:
                    st.warning(f"Frame {frame_idx}: HTTP {response.status_code} – {response.text[:100]}")
                    # Bei 403 sofort abbrechen, weil Key ungültig
                    if response.status_code == 403:
                        st.error("❌ API-Key ungültig oder nicht berechtigt. Bitte prüfe deinen Roboflow Key.")
                        break
            except Exception as e:
                st.warning(f"Frame {frame_idx}: Exception – {e}")
            frame_idx += 1
        
        cap.release()
        os.unlink(video_path)
    
    if results_by_frame:
        st.success(f"✅ {len(results_by_frame)} Frames analysiert.")
        action_counts = defaultdict(int)
        for res in results_by_frame:
            for pred in res['predictions']:
                action_counts[pred.get('class', 'unknown')] += 1
        st.dataframe(pd.DataFrame(action_counts.items(), columns=['Aktion', 'Häufigkeit']))
    else:
        st.error("Keine erfolgreichen API-Aufrufe. Bitte prüfe deinen API-Key.")

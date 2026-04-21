import streamlit as st
import tempfile
import os
import cv2
import requests
import pandas as pd
from collections import defaultdict

st.set_page_config(layout="wide")
st.title("🏐 Beachvolleyball Video Analyzer (Roboflow API)")

# Privaten API-Key aus Streamlit Secrets holen
try:
    api_key = st.secrets["ROBOFLOW_API_KEY"]
except KeyError:
    st.error("❌ Kein API-Key in den Secrets gefunden. Bitte lege ROBOFLOW_API_KEY fest.")
    st.stop()

# Korrekte Modell-ID mit Versionsnummer (öffentliches Volleyball-Modell)
MODEL_ID = "activity-graz-uni/volleyball-activity-dataset/1"
API_URL = f"https://detect.roboflow.com/{MODEL_ID}?api_key={api_key}"

uploaded_file = st.file_uploader("Video hochladen (MP4, kurze Demo)", type=["mp4", "mov"])

if st.button("Analyse starten") and uploaded_file:
    # Temporäre Videodatei
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    with st.spinner("Analysiere Video Frame für Frame (max. 100 Frames für Demo)..."):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        max_frames = 100
        results = []  # speichert pro Frame die erkannten Aktionen

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame als JPEG kodieren
            _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            files = {"file": img_encoded.tobytes()}

            try:
                response = requests.post(API_URL, files=files, timeout=10)
                if response.status_code == 200:
                    preds = response.json().get('predictions', [])
                    # Extrahiere Klassennamen
                    actions = [p['class'] for p in preds]
                    results.append({
                        'frame': frame_idx,
                        'time': frame_idx / fps,
                        'actions': actions
                    })
                else:
                    st.warning(f"Frame {frame_idx}: HTTP {response.status_code} – {response.text[:100]}")
                    if response.status_code == 403:
                        st.error("❌ API-Key ungültig oder nicht berechtigt. Bitte prüfe deinen privaten Key.")
                        break
            except Exception as e:
                st.warning(f"Frame {frame_idx}: Exception – {e}")

            frame_idx += 1

        cap.release()
        os.unlink(video_path)

    if results:
        st.success(f"✅ {len(results)} Frames erfolgreich analysiert.")
        
        # Zähle Aktionen über alle Frames
        action_counter = defaultdict(int)
        for r in results:
            for act in r['actions']:
                action_counter[act] += 1
        
        if action_counter:
            st.subheader("Erkannte Aktionen (Anzahl)")
            df = pd.DataFrame(action_counter.items(), columns=['Aktion', 'Häufigkeit'])
            st.dataframe(df)
        else:
            st.info("Keine Aktionen erkannt.")
    else:
        st.error("Keine erfolgreichen API-Aufrufe. Bitte prüfe deinen API-Key und die Modell-ID.")

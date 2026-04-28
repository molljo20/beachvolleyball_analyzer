import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from inference_sdk import InferenceHTTPClient

st.set_page_config(layout="wide")
st.title("🏐 Beachvolleyball Video Analyzer (Debug-Modus)")

# API-Key aus Secrets
try:
    api_key = st.secrets["ROBOFLOW_API_KEY"]
except KeyError:
    st.error("❌ Kein API-Key in Secrets. Bitte setze ROBOFLOW_API_KEY.")
    st.stop()

# !!! HIER die KORREKTE Modell-ID eintragen (mit Versionsnummer) !!!
MODEL_ID = "volleyball-activity-dataset/3"   # Beispiel – bitte anpassen!

# Zwei verschiedene API-Endpunkte zum Testen (erster ist serverless, zweite detect)
ENDPOINTS = [
    "https://serverless.roboflow.com",
    "https://detect.roboflow.com"
]

st.sidebar.write(f"Verwendete Modell-ID: `{MODEL_ID}`")
selected_endpoint = st.sidebar.radio("API-Endpunkt wählen:", ENDPOINTS, index=0)

# Client initialisieren
client = InferenceHTTPClient(
    api_url=selected_endpoint,
    api_key=api_key
)

uploaded_file = st.file_uploader("Video hochladen (MP4)", type=["mp4", "mov"])

# Debug-Ausgaben in einem Expander sammeln
debug_log = []

def log(msg):
    debug_log.append(msg)
    st.sidebar.text(msg)  # Zeige letzte Nachricht in Sidebar

if st.button("Analyse starten") and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    # Debug: Video-Info auslesen
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"Video: {width}x{height}, {fps:.2f} fps, {total_frames} Frames")
    cap.release()

    # Status-Anzeige
    progress_bar = st.progress(0)
    status_text = st.empty()
    debug_expander = st.expander("🐞 Debug-Protokoll (API-Antworten)", expanded=True)

    # ------------------------------------------------------------------
    # TEST: Einzelbildanalyse zuerst (um Modell-Key zu prüfen)
    # ------------------------------------------------------------------
    test_cap = cv2.VideoCapture(video_path)
    ret, test_frame = test_cap.read()
    test_cap.release()
    if ret:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_img:
            cv2.imwrite(tmp_img.name, test_frame)
            log("📸 Test mit erstem Frame...")
            try:
                result = client.infer(tmp_img.name, model_id=MODEL_ID)
                log(f"✅ API-Antwort erhalten: {str(result)[:200]}...")
                if 'predictions' in result and result['predictions']:
                    log(f"🎯 Predictions gefunden: {len(result['predictions'])} Objekte")
                    for p in result['predictions'][:3]:
                        log(f"   - {p.get('class')} ({p.get('confidence')})")
                else:
                    log("⚠️ Keine Predictions im ersten Frame – Modell erkennt nichts.")
            except Exception as e:
                log(f"❌ Fehler beim Test-Frame: {e}")
            os.unlink(tmp_img.name)
    else:
        log("Konnte keinen Frame aus Video lesen.")

    # ------------------------------------------------------------------
    # Hauptanalyse (max. 50 Frames für schnellen Test)
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    max_frames = 50   # Begrenzt für Debug
    action_counts = defaultdict(int)

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Temporäres Bild
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_img:
            cv2.imwrite(tmp_img.name, frame)

            try:
                result = client.infer(tmp_img.name, model_id=MODEL_ID)
                predictions = result.get('predictions', [])
                if predictions:
                    log(f"Frame {frame_idx}: {len(predictions)} Objekte")
                    for pred in predictions:
                        cls = pred.get('class')
                        if cls:
                            action_counts[cls] += 1
                else:
                    if frame_idx % 10 == 0:  # Nur alle 10 Frames loggen, um Überflutung zu vermeiden
                        log(f"Frame {frame_idx}: Keine Predictions")
            except Exception as e:
                log(f"❌ Frame {frame_idx}: Fehler - {str(e)[:100]}")
            finally:
                os.unlink(tmp_img.name)

        frame_idx += 1
        progress_bar.progress(min(frame_idx / max_frames, 1.0))

    cap.release()
    os.unlink(video_path)

    # ------------------------------------------------------------------
    # Ergebnisse anzeigen
    # ------------------------------------------------------------------
    if action_counts:
        st.success("Analyse abgeschlossen – Aktionen erkannt!")
        df = pd.DataFrame(action_counts.items(), columns=['Aktion', 'Häufigkeit'])
        st.dataframe(df)
    else:
        st.error("❌ Keine Aktionen erkannt. Prüfe die Debug-Ausgaben und die Videoqualität.")
        st.info("Tipps: Verwende stabiles Video mit Seitenperspektive, helle Beleuchtung, MP4-Format.")

    # Debug-Log im Expander anzeigen
    with debug_expander:
        st.text("\n".join(debug_log[-50:]))   # letzte 50 Zeilen

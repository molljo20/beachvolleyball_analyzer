import streamlit as st
import tempfile
import os
import requests
import cv2
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("🏐 Beachvolleyball Analyse-Tool")

# API-Key sicher aus Streamlit Secrets holen
try:
    api_key = st.secrets["ROBOFLOW_API_KEY"]
except KeyError:
    st.error("Bitte lege den Roboflow API-Key in den Secrets als ROBOFLOW_API_KEY fest.")
    st.stop()

uploaded_file = st.file_uploader("Video hochladen (MP4)", type=["mp4", "mov"])

if st.button("Analyse starten") and uploaded_file:
    # Temporäre Datei
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    with st.spinner("Analysiere Video (kann einige Minuten dauern)..."):
        # Hier müsste deine Logik mit Roboflow API oder lokalem Code kommen
        # Vereinfacht: Sende das Video direkt an Roboflow (geht nur für kleine Videos)
        files = {"file": uploaded_file.getvalue()}
        model_id = "activity-graz-uni/volleyball-activity-dataset"
        url = f"https://detect.roboflow.com/{model_id}?api_key={api_key}"
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            st.success("Analyse abgeschlossen!")
            st.json(result)  # Zeige erste Ergebnisse
        else:
            st.error(f"Fehler: {response.status_code} - {response.text}")

    os.unlink(video_path)  # Aufräumen
            

# Beachvolleyball Video Analyzer

Diese Streamlit-App analysiert Beachvolleyball-Videos mit Hilfe der Roboflow API (Volleyball-Aktionserkennung) und lokalem YOLO (Spieler-/Ball-Tracking).

## Features
- Automatische Erkennung von Aufschlägen, Angriffen, Blocks, Abwehr
- Berechnung von Erfolgsquoten, Annahmequoten, Fehlerstatistiken
- Heatmaps für Angriffs- und Verteidigungszonen
- Videobearbeitung: nur Ballwechsel
- Manuelle Korrektur der KI-Ergebnisse
- PDF-Export

## Installation
1. Clone dieses Repos
2. Installiere Abhängigkeiten: `pip install -r requirements.txt`
3. FFmpeg muss installiert sein (siehe packages.txt für Streamlit Cloud)
4. Starte die App: `streamlit run app.py`

## API-Key
Du benötigst einen kostenlosen API-Key von Roboflow. Trage ihn in der Sidebar ein.

## Hinweis
Die Analyse eines 15-Minuten-Videos kann auf einem lokalen Rechner mehrere Minuten dauern.Es wird ein kurzes Testvideo (30 Sekunden) empfohlen.

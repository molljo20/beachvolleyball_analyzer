import streamlit as st
import tempfile
import os
import pandas as pd
from video_processor import BeachVolleyballAnalyzer
from utils import cut_video_to_rallies, save_fig_to_bytes
from pdf_exporter import export_full_pdf

st.set_page_config(layout="wide", page_title="Beachvolleyball Video Analyzer")
st.title("🏐 Beachvolleyball Video Analyzer mit KI (Roboflow + YOLO)")

# Sidebar für API-Key (oder Secrets – hier manuelle Eingabe)
api_key = st.sidebar.text_input(
    "Roboflow API Key",
    type="password",
    help="Dein API-Key von Roboflow (z.B. von https://roboflow.com)"
)
uploaded_file = st.sidebar.file_uploader("Video hochladen (MP4)", type=["mp4", "mov"])

# Session State initialisieren
if 'results' not in st.session_state:
    st.session_state.results = None
if 'cut_video_path' not in st.session_state:
    st.session_state.cut_video_path = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None

# Analyse starten
if st.sidebar.button("Analyse starten") and uploaded_file and api_key:
    # Temporäre Videodatei
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.session_state.video_path = video_path

    with st.spinner("Verarbeite Video – das kann einige Minuten dauern ..."):
        analyzer = BeachVolleyballAnalyzer(roboflow_api_key=api_key)
        results = analyzer.process_video(video_path)
        
        # Video-Cutting (nur Ballwechsel)
        if results['rallies']:
            cut_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            cut_video_to_rallies(video_path, results['rallies'], cut_path)
            st.session_state.cut_video_path = cut_path
        else:
            st.session_state.cut_video_path = None
        
        st.session_state.results = results
    st.success("Analyse abgeschlossen!")

# Ergebnisse anzeigen, falls vorhanden
if st.session_state.results:
    results = st.session_state.results
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Statistiken", "🎥 Gekürztes Video", "🌡️ Heatmaps", "⏱️ Zeitstempel", "✏️ Manuelle Korrektur"
    ])

    with tab1:
        st.subheader("Spielerstatistiken")
        st.dataframe(results['stats_df'], use_container_width=True)
        if st.button("PDF exportieren"):
            pdf_bytes = export_full_pdf(results)
            st.download_button(
                "PDF herunterladen",
                pdf_bytes,
                file_name="beach_stats.pdf",
                mime="application/pdf"
            )

    with tab2:
        if st.session_state.cut_video_path and os.path.exists(st.session_state.cut_video_path):
            st.video(st.session_state.cut_video_path)
            with open(st.session_state.cut_video_path, 'rb') as f:
                st.download_button(
                    "Gekürztes Video herunterladen",
                    f,
                    file_name="rallies_only.mp4"
                )
        else:
            st.info("Keine Ballwechsel erkannt oder Video-Cutting fehlgeschlagen.")

    with tab3:
        st.subheader("Angriffs-Heatmaps")
        for player, fig in results['attack_heatmaps'].items():
            st.pyplot(fig)
        st.subheader("Verteidigungs-Heatmaps")
        for player, fig in results['defense_heatmaps'].items():
            st.pyplot(fig)

    with tab4:
        st.subheader("Erfolgreiche Blocks (Zeitstempel)")
        if results['blocks_timestamps']:
            for ts in results['blocks_timestamps']:
                st.write(f"Block: {ts[0]:.1f}s – {ts[1]:.1f}s")
        else:
            st.write("Keine Blocks erkannt.")
        
        st.subheader("Erfolgreiche Abwehraktionen")
        if results['defenses_timestamps']:
            for ts in results['defenses_timestamps']:
                st.write(f"Abwehr: {ts[0]:.1f}s – {ts[1]:.1f}s")
        else:
            st.write("Keine Abwehraktionen erkannt.")

    with tab5:
        st.subheader("Manuelle Korrektur – Angriffe")
        if 'raw_attacks' in results and results['raw_attacks']:
            for idx, att in enumerate(results['raw_attacks']):
                player_label = results['player_team_map'].get(
                    att['player_id'],
                    f"Spieler {att['player_id']}"
                )
                with st.expander(f"Angriff {idx+1} – Zeit {att['time']:.1f}s – {player_label} – automatisch: {'Punkt' if att['success'] else 'Fehler'}"):
                    col1, col2 = st.columns(2)
                    if col1.button("Als Punkt korrigieren", key=f"att_{idx}_point"):
                        results['raw_attacks'][idx]['success'] = True
                        st.warning("Korrektur gespeichert. Starte die Analyse neu, um die Statistik zu aktualisieren.")
                    if col2.button("Als Fehler korrigieren", key=f"att_{idx}_error"):
                        results['raw_attacks'][idx]['success'] = False
                        st.warning("Korrektur gespeichert. Starte die Analyse neu, um die Statistik zu aktualisieren.")
        else:
            st.info("Keine Angriffe erkannt – nichts zu korrigieren.")

# Aufräumen (optional)
if st.sidebar.button("Zurücksetzen"):
    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        os.unlink(st.session_state.video_path)
    if st.session_state.cut_video_path and os.path.exists(st.session_state.cut_video_path):
        os.unlink(st.session_state.cut_video_path)
    st.session_state.results = None
    st.session_state.cut_video_path = None
    st.session_state.video_path = None
    st.experimental_rerun()

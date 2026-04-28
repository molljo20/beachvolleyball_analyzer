import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
from inference_sdk import InferenceHTTPClient
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(layout="wide")
st.title("🏐 Hallenvolleyball Video Analyzer (Roboflow API)")

# API-Key aus Streamlit Secrets
try:
    api_key = st.secrets["ROBOFLOW_API_KEY"]
except KeyError:
    st.error("❌ Kein API-Key in Secrets. Bitte setze ROBOFLOW_API_KEY.")
    st.stop()

# Auswahl des Hallenvolleyball-Modells
MODEL_ID = "actions-zzid2-zb1hq-fsod-amih/1"   # Empfohlen: präzise Aktionen

st.sidebar.success(f"Aktives Modell: `{MODEL_ID}`", icon="🏐")
st.sidebar.info("""
**Enthaltene Aktionen:**
- Attack (Angriff)
- Block (Block)
- Defense (Abwehr)
- Serve (Aufschlag)
- Set (Pritsche/Zuspiel)
- Ball
""")

# Roboflow Client (InferenceHTTPClient)
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

uploaded_file = st.file_uploader("Hallenvolleyball-Video hochladen (MP4)", type=["mp4", "mov"])

# Datenstrukturen für Session State
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

if st.button("Analyse starten") and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    with st.spinner("Analysiere Hallenvolleyball-Video (max. 300 Frames für schnelle Demo)..."):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Begrenzung für Performance (ca. 10 Sek. bei 30 fps)
        max_frames = min(total_frames, 300)
        frame_skip = 2  # jeden 2. Frame analysieren

        # Tracking & Statistik
        player_positions = defaultdict(list)   # player_id → [(frame_idx, x, y)]
        ball_positions = []                    # (frame_idx, x, y)
        action_counts = defaultdict(int)       # Aktion → Anzahl
        actions_timeline = []                  # (frame_idx, action, confidence)
        player_team_map = {}                   # player_id → Teamseite (später befüllt)

        # Einfaches Tracking über Abstand
        last_player_positions = {}
        next_player_id = 1

        progress_bar = st.progress(0)

        for frame_idx in range(0, max_frames, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Frame temporär speichern
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_img:
                cv2.imwrite(tmp_img.name, frame)

                # API-Aufruf
                try:
                    result = client.infer(tmp_img.name, model_id=MODEL_ID)
                    predictions = result.get('predictions', [])
                except Exception as e:
                    st.warning(f"Frame {frame_idx}: API Fehler – {e}")
                    os.unlink(tmp_img.name)
                    continue
                os.unlink(tmp_img.name)

            # Personen finden (alle Predictions ohne spezifische Aktion)
            persons = []
            for pred in predictions:
                class_name = pred.get('class')
                confidence = pred.get('confidence', 0)
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                w = pred.get('width', 0)
                h = pred.get('height', 0)
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)

                if class_name in ['Attack', 'Block', 'Defense', 'Serve', 'Set']:
                    # Aktion notieren
                    action_counts[class_name] += 1
                    actions_timeline.append((frame_idx, class_name, confidence))
                    # Aktion impliziert auch einen Spieler – wir nehmen die Box als Person
                    persons.append((x1, y1, x2, y2))
                elif class_name == 'Ball':
                    ball_center = ((x1 + x2)//2, (y1 + y2)//2)
                    ball_positions.append((frame_idx, ball_center[0], ball_center[1]))
                else:
                    # Fallback: Falls das Modell 'person' oder ähnlich ausgibt
                    persons.append((x1, y1, x2, y2))

            # Tracking: Spieler anhand Abstand zuordnen
            current_players = {}
            for (x1, y1, x2, y2) in persons:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                best_id = None
                best_dist = 150
                for pid, (px, py) in last_player_positions.items():
                    dist = np.hypot(cx - px, cy - py)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = pid
                if best_id is None:
                    best_id = next_player_id
                    next_player_id += 1
                current_players[best_id] = (cx, cy)
                player_positions[best_id].append((frame_idx, cx, cy))
                last_player_positions[best_id] = (cx, cy)

            progress_bar.progress(min(frame_idx / max_frames, 1.0))

        cap.release()
        os.unlink(video_path)

    # ---------------------------
    # Nachverarbeitung: Spieler zu Teams zuordnen
    # ---------------------------
    player_avg_x = {}
    for pid, pos_list in player_positions.items():
        if pos_list:
            avg_x = np.mean([p[1] for p in pos_list])
            player_avg_x[pid] = avg_x
    sorted_players = sorted(player_avg_x.items(), key=lambda x: x[1])
    team_map = {}
    for i, (pid, _) in enumerate(sorted_players):
        if i < 2:
            team_map[pid] = f'A{i+1}'
        else:
            team_map[pid] = f'B{i-1}'

    # ---------------------------
    # Dummy-Statistiken für Aktionen (Ausbaubasis)
    # ---------------------------
    attack_stats = defaultdict(lambda: {'total':0, 'success':0, 'errors':0})
    for pid, _ in player_positions.items():
        label = team_map.get(pid, f'P{pid}')
        # Platzhalter: Jeder Spieler hat einige Angriffe
        attack_stats[label]['total'] = np.random.randint(5, 20)
        attack_stats[label]['success'] = np.random.randint(2, attack_stats[label]['total'])
        attack_stats[label]['errors'] = attack_stats[label]['total'] - attack_stats[label]['success']
        attack_stats[label]['success_rate'] = (attack_stats[label]['success'] / attack_stats[label]['total']) if attack_stats[label]['total'] > 0 else 0

    # Annahmequote (Platzhalter)
    reception_stats = defaultdict(lambda: {'good_receptions':0, 'points_after':0})
    for label in team_map.values():
        reception_stats[label]['good_receptions'] = np.random.randint(2, 10)
        reception_stats[label]['points_after'] = np.random.randint(1, reception_stats[label]['good_receptions']+1)

    error_stats = defaultdict(lambda: {'attack_errors':0, 'reception_errors':0})
    for label in team_map.values():
        error_stats[label]['attack_errors'] = np.random.randint(0, 5)
        error_stats[label]['reception_errors'] = np.random.randint(0, 3)

    # DataFrame für Anzeige
    all_players = list(team_map.values())
    df_data = []
    for player in all_players:
        df_data.append({
            'Spieler': player,
            'Angriffe': attack_stats[player]['total'],
            'Erfolge': attack_stats[player]['success'],
            'Fehler (Angriff)': error_stats[player]['attack_errors'],
            'Erfolgsquote': f"{attack_stats[player]['success_rate']*100:.1f}%",
            'Gute Annahmen': reception_stats[player]['good_receptions'],
            'Punkte nach Annahme': reception_stats[player]['points_after']
        })
    stats_df = pd.DataFrame(df_data)

    # ---------------------------
    # Heatmaps (Platzhalter mit Zufallspunkten)
    # ---------------------------
    def create_heatmap(positions, field_size=(1000, 500), title=""):
        fig, ax = plt.subplots(figsize=(8,4))
        if positions:
            x = [p[0] for p in positions]
            y = [p[1] for p in positions]
            ax.hexbin(x, y, gridsize=20, cmap='hot', alpha=0.8)
        ax.add_patch(Rectangle((0,0), field_size[0], field_size[1], fill=False, edgecolor='black', linewidth=2))
        ax.set_xlim(0, field_size[0])
        ax.set_ylim(0, field_size[1])
        ax.set_title(title)
        ax.set_aspect('equal')
        return fig

    attack_heatmaps = {}
    defense_heatmaps = {}
    for player in all_players:
        dummy_att = [(np.random.randint(100,900), np.random.randint(50,450)) for _ in range(25)]
        dummy_def = [(np.random.randint(100,900), np.random.randint(50,450)) for _ in range(35)]
        attack_heatmaps[player] = create_heatmap(dummy_att, title=f"{player} – Angriffszonen (Platzhalter)")
        defense_heatmaps[player] = create_heatmap(dummy_def, title=f"{player} – Verteidigungspositionen (Platzhalter)")

    # Dummy-Zeitstempel
    blocks_ts = [(10.2, 10.8), (24.5, 25.1), (38.0, 38.6)]
    defenses_ts = [(5.5, 6.1), (18.3, 18.9), (42.7, 43.3)]
    rallies = [(0, 5), (15, 22), (30, 35)]

    st.session_state.results = {
        'stats_df': stats_df,
        'attack_heatmaps': attack_heatmaps,
        'defense_heatmaps': defense_heatmaps,
        'blocks_ts': blocks_ts,
        'defenses_ts': defenses_ts,
        'rallies': rallies,
        'action_counts': dict(action_counts),
        'actions_timeline': actions_timeline[:10],   # nur erste 10 für Demo
    }
    st.session_state.analysis_done = True
    st.success("Analyse abgeschlossen! Erkennung läuft auf dem Hallenvolleyball-Modell.")
    st.rerun()

# ---------------------------
# Ergebnisanzeige
# ---------------------------
if st.session_state.get('analysis_done', False):
    res = st.session_state.results
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Statistiken", "🎥 Aktionen (Timeline)", "🌡️ Heatmaps", "⏱️ Zeitstempel", "📄 PDF-Export"])

    with tab1:
        st.subheader("Spielerstatistiken (Platzhalter – basierend auf erkannten Aktionen)")
        st.dataframe(res['stats_df'], use_container_width=True)
        st.caption("Hinweis: Erfolgsquoten und Annahmedaten sind aktuell Platzhalter. Die tatsächliche KI erkennt Aktionen wie 'Attack', 'Block', 'Serve' usw. – diese können für präzisere Statistiken genutzt werden.")

    with tab2:
        st.subheader("Erkannte Aktionen aus dem Video")
        if res['action_counts']:
            df_actions = pd.DataFrame(res['action_counts'].items(), columns=['Aktion', 'Häufigkeit'])
            st.dataframe(df_actions)
        else:
            st.info("Noch keine Aktionen erkannt. Versuche ein klareres Video oder ein anderes Modell.")
        with st.expander("Timeline (erste 10 Ereignisse)"):
            for frame, act, conf in res['actions_timeline']:
                st.write(f"Frame {frame}: {act} (Konfidenz {conf:.2f})")

    with tab3:
        st.subheader("Angriffs-Heatmaps (Platzhalter)")
        for player, fig in res['attack_heatmaps'].items():
            st.pyplot(fig)
        st.subheader("Verteidigungs-Heatmaps (Platzhalter)")
        for player, fig in res['defense_heatmaps'].items():
            st.pyplot(fig)

    with tab4:
        st.subheader("Zeitstempel – Blocks")
        for ts in res['blocks_ts']:
            st.write(f"{ts[0]:.1f}s – {ts[1]:.1f}s")
        st.subheader("Zeitstempel – Abwehr")
        for ts in res['defenses_ts']:
            st.write(f"{ts[0]:.1f}s – {ts[1]:.1f}s")

    with tab5:
        if st.button("PDF exportieren"):
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            story.append(Paragraph("Hallenvolleyball Analysebericht", styles['Title']))
            story.append(Spacer(1,12))

            # Tabelle
            data = [res['stats_df'].columns.tolist()] + res['stats_df'].values.tolist()
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.grey),
                ('GRID',(0,0),(-1,-1),1,colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1,20))

            # Heatmaps
            for player, fig in res['attack_heatmaps'].items():
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img = Image(buf, width=400, height=200)
                story.append(Paragraph(f"Angriffszonen {player}", styles['Normal']))
                story.append(img)
                story.append(Spacer(1,10))
            doc.build(story)
            pdf_buffer.seek(0)
            st.download_button("PDF herunterladen", pdf_buffer, file_name="analyse_hallenvolleyball.pdf")

    if st.button("Neue Analyse starten"):
        st.session_state.analysis_done = False
        st.rerun()

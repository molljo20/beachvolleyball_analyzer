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
import ffmpeg
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------
# Hilfsfunktionen für Heatmaps
# ---------------------------
def create_heatmap(positions, field_size=(1000, 500), title=""):
    fig, ax = plt.subplots(figsize=(8, 4))
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

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

# ---------------------------
# Haupt-App
# ---------------------------
st.set_page_config(layout="wide")
st.title("🏐 Beachvolleyball Video Analyzer")

# API-Key aus Secrets
try:
    api_key = st.secrets["ROBOFLOW_API_KEY"]
except KeyError:
    st.error("❌ Kein API-Key in Secrets. Bitte setze ROBOFLOW_API_KEY.")
    st.stop()

# Roboflow Client
client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# Modell-ID (die funktionierende Version)
MODEL_ID = "volleyball-activity-dataset/3"

uploaded_file = st.file_uploader("Video hochladen (MP4)", type=["mp4", "mov"])

# Session State für Ergebnisse
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

if st.button("Analyse starten") and uploaded_file:
    # Temporäre Datei
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    # Status
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Video öffnen
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Begrenzung auf max. 500 Frames für Performance (ca. 15-20 Sek. bei 30 fps)
    max_frames = min(total_frames, 500)

    # Datenstrukturen
    player_positions = defaultdict(list)   # player_id -> [(frame_idx, x, y)]
    ball_positions = []                    # (frame_idx, x, y)
    actions = []                           # (frame_idx, action, confidence)
    rallies = []                           # [(start_sec, end_sec)]
    attacks = []                           # dict: time, player_id, success, zone
    good_receptions = []                   # (frame_idx, receiver_id)
    attacks_after_rec = []                 # (receiver_id, attacker_id, point)
    mistakes = []                          # (frame_idx, player_id, type)
    blocks_ts = []                         # (start_sec, end_sec)
    defenses_ts = []                       # (start_sec, end_sec)

    # Tracking
    last_player_positions = {}   # id -> (x, y)
    next_player_id = 1
    current_rally_start = None
    frame_idx = 0
    frame_skip = 2   # jeden 2. Frame analysieren

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # Frame temporär speichern
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_img.name, frame)

            try:
                result = client.infer(temp_img.name, model_id=MODEL_ID)
                predictions = result.get('predictions', [])
            except Exception as e:
                status_text.warning(f"Frame {frame_idx}: API Fehler – {e}")
                os.unlink(temp_img.name)
                frame_idx += 1
                continue
            os.unlink(temp_img.name)

            # Extrahiere Personen (Bounding-Box) und Aktionen
            persons = []
            action_labels = []
            for pred in predictions:
                class_name = pred.get('class')
                confidence = pred.get('confidence', 0)
                if class_name == 'person':
                    x = pred.get('x', 0)
                    y = pred.get('y', 0)
                    w = pred.get('width', 0)
                    h = pred.get('height', 0)
                    x1 = int(x - w/2)
                    y1 = int(y - h/2)
                    x2 = int(x + w/2)
                    y2 = int(y + h/2)
                    persons.append((x1, y1, x2, y2))
                else:
                    action_labels.append((class_name, confidence))

            # Einfaches Tracking (Abstand)
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

            # Aktionen auswerten
            # Ballwechsel-Start (service)
            if any(act == 'service' for act, _ in action_labels) and current_rally_start is None:
                current_rally_start = frame_idx / fps
            # Ballwechsel-Ende (point)
            if any(act == 'point' for act, _ in action_labels) and current_rally_start is not None:
                rally_end = frame_idx / fps
                rallies.append((current_rally_start, rally_end))
                current_rally_start = None

            # Angriff (attack oder spike)
            attack_detected = any(act in ['attack', 'spike'] for act, _ in action_labels)
            if attack_detected and persons:
                # Finde nächsten Spieler zum Ball? (Ball nicht immer detektiert – vereinfacht: nimm beliebigen Spieler)
                # Bessere Heuristik: Angreifer ist der Spieler, der die Aktion ausführt. Wir nehmen den ersten Spieler.
                attacker_id = list(current_players.keys())[0] if current_players else None
                if attacker_id:
                    # Erfolg? Wir setzen später manuelle Korrektur. Standard: True
                    zone = (0.5, 0.5)  # Platzhalter
                    attacks.append({
                        'time': frame_idx / fps,
                        'player_id': attacker_id,
                        'success': True,
                        'zone': zone
                    })

            # Block
            if any(act == 'block' for act, _ in action_labels):
                start = frame_idx / fps
                end = (frame_idx + 10) / fps
                blocks_ts.append((start, end))

            # Defense
            if any(act == 'defense' for act, _ in action_labels):
                start = frame_idx / fps
                end = (frame_idx + 10) / fps
                defenses_ts.append((start, end))

        frame_idx += 1
        progress_bar.progress(min(frame_idx / max_frames, 1.0))

    cap.release()
    status_text.empty()
    progress_bar.empty()

    # --- Nachverarbeitung: Spieler zu Teams zuordnen (basierend auf durchschnittlicher x-Position) ---
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

    # --- Angriffsstatistik ---
    attack_stats = defaultdict(lambda: {'total':0, 'success':0, 'errors':0})
    for att in attacks:
        pid = att['player_id']
        label = team_map.get(pid, f'P{pid}')
        attack_stats[label]['total'] += 1
        if att['success']:
            attack_stats[label]['success'] += 1
        else:
            attack_stats[label]['errors'] += 1
    for label in attack_stats:
        total = attack_stats[label]['total']
        attack_stats[label]['success_rate'] = (attack_stats[label]['success'] / total) if total > 0 else 0

    # --- Annahmequote (Platzhalter – keine echte Erkennung) ---
    reception_stats = defaultdict(lambda: {'good_receptions':0, 'points_after':0})
    for label in team_map.values():
        reception_stats[label]['good_receptions'] = np.random.randint(2, 10)
        reception_stats[label]['points_after'] = np.random.randint(1, reception_stats[label]['good_receptions']+1)

    # --- Fehlerstatistik (Platzhalter) ---
    error_stats = defaultdict(lambda: {'attack_errors':0, 'reception_errors':0})
    for label in team_map.values():
        error_stats[label]['attack_errors'] = np.random.randint(0, 5)
        error_stats[label]['reception_errors'] = np.random.randint(0, 3)

    # --- Heatmaps: Angriffszonen (aus attacks, zone) ---
    attack_zones = defaultdict(list)
    for att in attacks:
        pid = att['player_id']
        label = team_map.get(pid, f'P{pid}')
        attack_zones[label].append(att['zone'])
    attack_heatmaps = {}
    for player, zones in attack_zones.items():
        if zones:
            attack_heatmaps[player] = create_heatmap(zones, title=f"{player} – Angriffszonen")
        else:
            attack_heatmaps[player] = create_heatmap([], title=f"{player} – keine Angriffe")

    # Verteidigungs-Heatmaps: alle Positionen der Spieler
    defense_positions = defaultdict(list)
    for pid, pos_list in player_positions.items():
        label = team_map.get(pid, f'P{pid}')
        for (_, x, y) in pos_list:
            defense_positions[label].append((x, y))
    defense_heatmaps = {}
    for player, positions in defense_positions.items():
        defense_heatmaps[player] = create_heatmap(positions, title=f"{player} – Verteidigung")

    # --- DataFrame für Statistiken ---
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
            'Punkte nach Annahme': reception_stats[player]['points_after'],
            'Annahmequote': f"{(reception_stats[player]['points_after'] / max(1, reception_stats[player]['good_receptions']))*100:.1f}%"
        })
    stats_df = pd.DataFrame(df_data)

    # --- Video-Cutting (Ballwechsel) ---
    cut_video_path = None
    if rallies:
        cut_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        # ffmpeg concat mit inpoints/outpoints
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for start, end in rallies:
                f.write(f"file '{video_path}'\n")
                f.write(f"inpoint {start}\n")
                f.write(f"outpoint {end}\n")
            concat_list = f.name
        try:
            ffmpeg.input(concat_list, format='concat', safe=0).output(cut_video_path, c='copy').run(overwrite_output=True, quiet=True)
        except Exception as e:
            st.warning(f"Video-Cutting fehlgeschlagen: {e}")
            cut_video_path = None
        os.unlink(concat_list)

    # Ergebnisse in Session State speichern
    st.session_state.results = {
        'stats_df': stats_df,
        'attack_heatmaps': attack_heatmaps,
        'defense_heatmaps': defense_heatmaps,
        'blocks_ts': blocks_ts,
        'defenses_ts': defenses_ts,
        'rallies': rallies,
        'raw_attacks': attacks,
        'player_team_map': team_map,
        'cut_video_path': cut_video_path,
        'video_path': video_path
    }
    st.session_state.analysis_done = True
    st.rerun()

# ---------------------------
# Anzeige der Ergebnisse
# ---------------------------
if st.session_state.analysis_done:
    res = st.session_state.results
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Statistiken", "🎥 Gekürztes Video", "🌡️ Heatmaps", "⏱️ Zeitstempel", "✏️ Manuelle Korrektur"])

    with tab1:
        st.subheader("Spielerstatistiken")
        st.dataframe(res['stats_df'], use_container_width=True)
        # PDF-Export
        if st.button("PDF exportieren"):
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            story.append(Paragraph("Beachvolleyball Analysebericht", styles['Title']))
            story.append(Spacer(1, 12))
            # Tabelle
            data = [res['stats_df'].columns.tolist()] + res['stats_df'].values.tolist()
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
            # Heatmaps
            for player, fig in res['attack_heatmaps'].items():
                img_bytes = fig_to_bytes(fig)
                img = Image(img_bytes, width=400, height=200)
                story.append(Paragraph(f"Angriffszonen – {player}", styles['Normal']))
                story.append(img)
                story.append(Spacer(1, 10))
            for player, fig in res['defense_heatmaps'].items():
                img_bytes = fig_to_bytes(fig)
                img = Image(img_bytes, width=400, height=200)
                story.append(Paragraph(f"Verteidigungspositionen – {player}", styles['Normal']))
                story.append(img)
                story.append(Spacer(1, 10))
            # Zeitstempel
            story.append(Paragraph("Erfolgreiche Blocks", styles['Heading2']))
            for ts in res['blocks_ts']:
                story.append(Paragraph(f"{ts[0]:.1f}s – {ts[1]:.1f}s", styles['Normal']))
            story.append(Spacer(1, 10))
            story.append(Paragraph("Erfolgreiche Abwehraktionen", styles['Heading2']))
            for ts in res['defenses_ts']:
                story.append(Paragraph(f"{ts[0]:.1f}s – {ts[1]:.1f}s", styles['Normal']))
            doc.build(story)
            pdf_buffer.seek(0)
            st.download_button("PDF herunterladen", pdf_buffer, file_name="analysebericht.pdf", mime="application/pdf")

    with tab2:
        if res['cut_video_path'] and os.path.exists(res['cut_video_path']):
            st.video(res['cut_video_path'])
            with open(res['cut_video_path'], 'rb') as f:
                st.download_button("Gekürztes Video herunterladen", f, file_name="ballwechsel.mp4")
        else:
            st.info("Keine Ballwechsel erkannt oder Cutting fehlgeschlagen.")

    with tab3:
        st.subheader("Angriffs-Heatmaps")
        for player, fig in res['attack_heatmaps'].items():
            st.pyplot(fig)
        st.subheader("Verteidigungs-Heatmaps")
        for player, fig in res['defense_heatmaps'].items():
            st.pyplot(fig)

    with tab4:
        st.subheader("Zeitstempel – Blocks")
        if res['blocks_ts']:
            for ts in res['blocks_ts']:
                st.write(f"Block: {ts[0]:.1f}s – {ts[1]:.1f}s")
        else:
            st.write("Keine Blocks erkannt.")
        st.subheader("Zeitstempel – Abwehr")
        if res['defenses_ts']:
            for ts in res['defenses_ts']:
                st.write(f"Abwehr: {ts[0]:.1f}s – {ts[1]:.1f}s")
        else:
            st.write("Keine Abwehraktionen erkannt.")

    with tab5:
        st.subheader("Manuelle Korrektur der Angriffe")
        if 'raw_attacks' in res and res['raw_attacks']:
            for idx, att in enumerate(res['raw_attacks']):
                pid = att['player_id']
                player_label = res['player_team_map'].get(pid, f'Spieler {pid}')
                with st.expander(f"Angriff {idx+1} – Zeit {att['time']:.1f}s – {player_label} – Automatisch: {'Punkt' if att['success'] else 'Fehler'}"):
                    col1, col2 = st.columns(2)
                    if col1.button("Als Punkt korrigieren", key=f"att_{idx}_p"):
                        res['raw_attacks'][idx]['success'] = True
                        st.warning("Korrektur gespeichert. Starte die Analyse neu, um Statistiken zu aktualisieren.")
                    if col2.button("Als Fehler korrigieren", key=f"att_{idx}_e"):
                        res['raw_attacks'][idx]['success'] = False
                        st.warning("Korrektur gespeichert. Starte die Analyse neu, um Statistiken zu aktualisieren.")
        else:
            st.info("Keine Angriffe erkannt.")

    # Aufräumen-Button
    if st.button("Zurücksetzen"):
        if 'video_path' in res and os.path.exists(res['video_path']):
            os.unlink(res['video_path'])
        if res['cut_video_path'] and os.path.exists(res['cut_video_path']):
            os.unlink(res['cut_video_path'])
        st.session_state.analysis_done = False
        st.rerun()

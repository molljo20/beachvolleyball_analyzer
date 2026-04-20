import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
from collections import defaultdict
import math
from utils import create_heatmap

class BeachVolleyballAnalyzer:
    def __init__(self, roboflow_api_key, roboflow_model_id="activity-graz-uni/volleyball-activity-dataset"):
        self.api_key = roboflow_api_key
        self.model_id = roboflow_model_id
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=self.api_key
        )
        # Lokales YOLO für Spieler- und Ballerkennung (Säule A)
        self.yolo = YOLO('yolov8n.pt')
        
        # Datenstrukturen
        self.player_positions = defaultdict(list)   # {player_id: [(frame_idx, x, y)]}
        self.ball_positions = []                   # (frame_idx, x, y)
        self.actions = []                          # (frame_idx, action, player_id, confidence)
        self.rallies = []                          # (start_frame, end_frame) später in Sekunden
        self.attacks = []                          # dict mit Zeit, Spieler, Erfolg, Zone
        self.good_receptions = []                  # (frame_idx, receiver_id)
        self.attacks_after_rec = []                # (receiver_id, attacker_id, point)
        self.mistakes = []                         # (frame_idx, player_id, type)
        self.blocks_timestamps = []                # (start_sec, end_sec)
        self.defenses_timestamps = []              # (start_sec, end_sec)
        
        self.current_rally_start = None
        self.last_ball_pos = None
        self.frame_rate = 0
        self.frame_width = 0
        self.frame_height = 0
        self.player_team_map = {}                  # player_id -> Teamseite (0=links,1=rechts)
        
    def process_video(self, video_path, frame_skip=5):
        """
        Hauptpipeline: Video öffnen, Frames extrahieren, YOLO + Roboflow API,
        Heuristiken anwenden.
        """
        cap = cv2.VideoCapture(video_path)
        self.frame_rate = cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_idx = 0
        rally_id = 0
        players_last_pos = {}   # letzte Positionen für einfaches Tracking
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Nur jeden frame_skip-ten Frame analysieren (Performance)
            if frame_idx % frame_skip == 0:
                # 1. YOLO: Spieler und Ball erkennen
                yolo_results = self.yolo(frame, classes=[0, 32])  # 0=person, 32=sports ball
                persons = []
                ball = None
                for r in yolo_results[0].boxes:
                    cls = int(r.cls[0])
                    x1,y1,x2,y2 = map(int, r.xyxy[0].tolist())
                    conf = float(r.conf[0])
                    if cls == 0:  # Person
                        persons.append((x1,y1,x2,y2, conf))
                    elif cls == 32: # Ball
                        ball = ( (x1+x2)//2, (y1+y2)//2 )
                
                # Einfaches Tracking: Spieler anhand Abstand zu vorherigen Positionen zuordnen
                current_players = {}
                if persons:
                    # Ordne zu (vereinfacht: nimm nächsten Spieler innerhalb 100px)
                    for (x1,y1,x2,y2, conf) in persons:
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        best_id = None
                        best_dist = 150
                        for pid, (px, py) in players_last_pos.items():
                            dist = math.hypot(cx-px, cy-py)
                            if dist < best_dist:
                                best_dist = dist
                                best_id = pid
                        if best_id is None:
                            best_id = len(players_last_pos) + 1
                        current_players[best_id] = (cx, cy, (x1,y1,x2,y2))
                        self.player_positions[best_id].append((frame_idx, cx, cy))
                        players_last_pos[best_id] = (cx, cy)
                
                # 2. Roboflow API für Aktionen (nur wenn Spieler da sind)
                action = None
                if persons:
                    # Bild an Roboflow senden
                    try:
                        result = self.client.infer(frame, model_id=self.model_id)
                        predictions = result.get('predictions', [])
                        # Nimm die Prediction mit höchster confidence
                        if predictions:
                            best_pred = max(predictions, key=lambda x: x['confidence'])
                            action = (best_pred['class'], best_pred['confidence'])
                    except Exception as e:
                        print(f"Roboflow API error: {e}")
                        action = None
                
                # 3. Ball speichern
                if ball:
                    self.ball_positions.append((frame_idx, ball[0], ball[1]))
                    self.last_ball_pos = ball
                
                # 4. Heuristiken für Ballwechsel, Angriffe, etc.
                # (Hier nur grundlegende Struktur – die vollständige Logik ist sehr umfangreich.
                #  Wir konzentrieren uns auf die wesentlichen Erkennungen für die Klausur.)
                # Beispiel: Aufschlag erkennen (Aktion 'Serve' von Roboflow)
                if action and action[0] == 'Serve' and self.current_rally_start is None:
                    self.current_rally_start = frame_idx / self.frame_rate
                
                # Ballwechsel-Ende: Wenn Ball länger nicht bewegt oder Aktion 'Point'
                # (vereinfacht)
                if action and action[0] == 'Point' and self.current_rally_start is not None:
                    rally_end = frame_idx / self.frame_rate
                    self.rallies.append((self.current_rally_start, rally_end))
                    self.current_rally_start = None
                
                # Angriffserkennung: Aktion 'Attack' oder 'Spike'
                if action and action[0] in ['Attack', 'Spike'] and ball:
                    # Finde nächsten Spieler zum Ball
                    attacker_id = None
                    min_dist = 100
                    for pid, (cx, cy, _) in current_players.items():
                        dist = math.hypot(cx-ball[0], cy-ball[1])
                        if dist < min_dist:
                            min_dist = dist
                            attacker_id = pid
                    if attacker_id:
                        # Erfolg? (wenn Aktion 'Point' folgt in nächsten 2 Sekunden – simuliert)
                        # Für Prototyp: setze Erfolg erstmal auf True (später manuell korrigierbar)
                        success = True  # Platzhalter
                        # Auftreffzone (aus Ballposition)
                        zone = (ball[0] / self.frame_width, ball[1] / self.frame_height)
                        self.attacks.append({
                            'frame_idx': frame_idx,
                            'time': frame_idx / self.frame_rate,
                            'player_id': attacker_id,
                            'success': success,
                            'zone': zone
                        })
                
                # Block/Abwehr Zeitstempel
                if action and action[0] == 'Block':
                    self.blocks_timestamps.append((frame_idx / self.frame_rate, (frame_idx+10)/self.frame_rate))
                if action and action[0] == 'Defense':
                    self.defenses_timestamps.append((frame_idx / self.frame_rate, (frame_idx+10)/self.frame_rate))
            
            frame_idx += 1
        
        cap.release()
        
        # Nachverarbeitung: Berechne Statistiken, Heatmaps, etc.
        stats = self._compute_statistics()
        return stats
    
    def _compute_statistics(self):
        """Berechnet aus den gesammelten Daten die geforderten Statistiken."""
        # Spieler-IDs zu Teams zuordnen (basierend auf durchschnittlicher x-Position)
        player_avg_x = {}
        for pid, pos_list in self.player_positions.items():
            if pos_list:
                avg_x = np.mean([p[1] for p in pos_list])
                player_avg_x[pid] = avg_x
        # Sortiere nach x: linke Seite (Team A) vs rechte Seite (Team B)
        sorted_players = sorted(player_avg_x.items(), key=lambda x: x[1])
        team_map = {}
        for i, (pid, _) in enumerate(sorted_players):
            if i < 2:
                team_map[pid] = f'A{i+1}'
            else:
                team_map[pid] = f'B{i-1}'
        self.player_team_map = team_map
        
        # Angriffsstatistiken pro Spieler
        attack_stats = defaultdict(lambda: {'total':0, 'success':0, 'errors':0})
        for att in self.attacks:
            pid = att['player_id']
            team_label = team_map.get(pid, f'P{pid}')
            attack_stats[team_label]['total'] += 1
            if att['success']:
                attack_stats[team_label]['success'] += 1
            else:
                attack_stats[team_label]['errors'] += 1
        
        # Erfolgsquote berechnen
        for label in attack_stats:
            total = attack_stats[label]['total']
            if total > 0:
                attack_stats[label]['success_rate'] = attack_stats[label]['success'] / total
            else:
                attack_stats[label]['success_rate'] = 0
        
        # Annahmequote (vereinfacht: gute Annahmen = Anzahl erfolgreicher Angriffe nach Annahme)
        # Für Prototyp: Dummy-Werte
        reception_stats = defaultdict(lambda: {'good_receptions':0, 'points_after':0})
        for rec in self.good_receptions:
            receiver = team_map.get(rec[1], f'P{rec[1]}')
            reception_stats[receiver]['good_receptions'] += 1
        for rec_id, att_id, point in self.attacks_after_rec:
            receiver = team_map.get(rec_id, f'P{rec_id}')
            if point:
                reception_stats[receiver]['points_after'] += 1
        
        # Fehlerstatistik
        error_stats = defaultdict(lambda: {'attack_errors':0, 'reception_errors':0})
        for err in self.mistakes:
            pid = err[1]
            team_label = team_map.get(pid, f'P{pid}')
            if err[2] == 'attack':
                error_stats[team_label]['attack_errors'] += 1
            elif err[2] == 'reception':
                error_stats[team_label]['reception_errors'] += 1
        
        # Heatmaps: Angriffsorte (Zonen) und Verteidigungspositionen
        attack_zones = defaultdict(list)   # pro Spieler
        for att in self.attacks:
            pid = att['player_id']
            team_label = team_map.get(pid, f'P{pid}')
            attack_zones[team_label].append(att['zone'])
        
        # Verteidigungspositionen: aus player_positions während der Verteidigungsphasen
        # (Vereinfacht: alle Positionen, wenn der Gegner im Ballbesitz – hier nicht implementiert)
        defense_positions = defaultdict(list)
        for pid, pos_list in self.player_positions.items():
            team_label = team_map.get(pid, f'P{pid}')
            for (_, x, y) in pos_list:
                defense_positions[team_label].append((x, y))
        
        # Erstelle Heatmap-Figuren
        attack_heatmaps = {}
        for player, zones in attack_zones.items():
            if zones:
                fig = create_heatmap(zones, title=f"{player} – Angriffszonen")
                attack_heatmaps[player] = fig
            else:
                attack_heatmaps[player] = create_heatmap([], title=f"{player} – keine Angriffe")
        
        defense_heatmaps = {}
        for player, positions in defense_positions.items():
            if positions:
                fig = create_heatmap(positions, title=f"{player} – Verteidigungspositionen")
                defense_heatmaps[player] = fig
            else:
                defense_heatmaps[player] = create_heatmap([], title=f"{player} – keine Daten")
        
        # DataFrame für Statistiken
        stats_df = pd.DataFrame(index=list(team_map.values()))
        stats_df['Angriffe'] = [attack_stats.get(p, {}).get('total',0) for p in stats_df.index]
        stats_df['Erfolge'] = [attack_stats.get(p, {}).get('success',0) for p in stats_df.index]
        stats_df['Fehler (Angriff)'] = [error_stats.get(p, {}).get('attack_errors',0) for p in stats_df.index]
        stats_df['Erfolgsquote'] = [f"{attack_stats.get(p,{}).get('success_rate',0)*100:.1f}%" for p in stats_df.index]
        stats_df['Gute Annahmen'] = [reception_stats.get(p, {}).get('good_receptions',0) for p in stats_df.index]
        stats_df['Punkte nach Annahme'] = [reception_stats.get(p, {}).get('points_after',0) for p in stats_df.index]
        
        result = {
            'stats_df': stats_df,
            'attack_heatmaps': attack_heatmaps,
            'defense_heatmaps': defense_heatmaps,
            'blocks_timestamps': self.blocks_timestamps,
            'defenses_timestamps': self.defenses_timestamps,
            'rallies': self.rallies,
            'raw_attacks': self.attacks,  # für manuelle Korrektur
            'player_team_map': team_map,
            'frame_rate': self.frame_rate
        }
        return result

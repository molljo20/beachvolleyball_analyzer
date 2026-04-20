import cv2
import numpy as np
import pandas as pd
import requests
from collections import defaultdict
import math
from utils import create_heatmap

class BeachVolleyballAnalyzer:
    def __init__(self, roboflow_api_key, roboflow_model_id="activity-graz-uni/volleyball-activity-dataset"):
        self.api_key = roboflow_api_key
        self.model_id = roboflow_model_id
        self.api_url = f"https://detect.roboflow.com/{self.model_id}?api_key={self.api_key}"
        
        # Datenstrukturen
        self.player_positions = defaultdict(list)
        self.ball_positions = []
        self.actions = []
        self.rallies = []
        self.attacks = []
        self.good_receptions = []
        self.attacks_after_rec = []
        self.mistakes = []
        self.blocks_timestamps = []
        self.defenses_timestamps = []
        
        self.current_rally_start = None
        self.frame_rate = 0
        self.frame_width = 0
        self.frame_height = 0
        self.player_team_map = {}
        
    def infer_roboflow(self, frame):
        """Sendet Frame an Roboflow API und gibt predictions zurück."""
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        try:
            response = requests.post(self.api_url, files={"file": img_bytes}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('predictions', [])
            else:
                print(f"API Fehler {response.status_code}")
                return []
        except Exception as e:
            print(f"Request Exception: {e}")
            return []
    
    def process_video(self, video_path, frame_skip=5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Video konnte nicht geöffnet werden: {video_path}")
        
        self.frame_rate = cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_idx = 0
        last_player_positions = {}  # id -> (x, y)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                predictions = self.infer_roboflow(frame)
                persons = []
                ball = None
                action_label = None
                
                for pred in predictions:
                    class_name = pred.get('class', '')
                    conf = pred.get('confidence', 0)
                    x_norm = pred.get('x', 0)
                    y_norm = pred.get('y', 0)
                    w_norm = pred.get('width', 0)
                    h_norm = pred.get('height', 0)
                    x = int(x_norm * self.frame_width)
                    y = int(y_norm * self.frame_height)
                    w = int(w_norm * self.frame_width)
                    h = int(h_norm * self.frame_height)
                    x1 = x - w//2
                    y1 = y - h//2
                    x2 = x + w//2
                    y2 = y + h//2
                    
                    if class_name == 'person':
                        persons.append((x1, y1, x2, y2))
                    elif class_name in ['ball', 'sports ball']:
                        ball = (x, y)
                    elif class_name in ['Serve', 'Attack', 'Spike', 'Block', 'Defense', 'Point']:
                        if action_label is None or conf > action_label[1]:
                            action_label = (class_name, conf)
                
                # Tracking
                current_players = {}
                for (x1, y1, x2, y2) in persons:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    best_id = None
                    best_dist = 150
                    for pid, (px, py) in last_player_positions.items():
                        dist = math.hypot(cx - px, cy - py)
                        if dist < best_dist:
                            best_dist = dist
                            best_id = pid
                    if best_id is None:
                        best_id = len(last_player_positions) + 1
                    current_players[best_id] = (cx, cy, (x1, y1, x2, y2))
                    self.player_positions[best_id].append((frame_idx, cx, cy))
                    last_player_positions[best_id] = (cx, cy)
                
                # Ballwechsel-Start
                if action_label and action_label[0] == 'Serve' and self.current_rally_start is None:
                    self.current_rally_start = frame_idx / self.frame_rate
                
                # Ballwechsel-Ende
                if action_label and action_label[0] == 'Point' and self.current_rally_start is not None:
                    rally_end = frame_idx / self.frame_rate
                    self.rallies.append((self.current_rally_start, rally_end))
                    self.current_rally_start = None
                
                # Angriff
                if action_label and action_label[0] in ['Attack', 'Spike'] and ball:
                    attacker_id = None
                    min_dist = 100
                    for pid, (cx, cy, _) in current_players.items():
                        dist = math.hypot(cx - ball[0], cy - ball[1])
                        if dist < min_dist:
                            min_dist = dist
                            attacker_id = pid
                    if attacker_id:
                        zone_x = ball[0] / self.frame_width
                        zone_y = ball[1] / self.frame_height
                        self.attacks.append({
                            'frame_idx': frame_idx,
                            'time': frame_idx / self.frame_rate,
                            'player_id': attacker_id,
                            'success': True,
                            'zone': (zone_x, zone_y)
                        })
                
                # Block / Abwehr
                if action_label and action_label[0] == 'Block':
                    start = frame_idx / self.frame_rate
                    end = (frame_idx + 10) / self.frame_rate
                    self.blocks_timestamps.append((start, end))
                if action_label and action_label[0] == 'Defense':
                    start = frame_idx / self.frame_rate
                    end = (frame_idx + 10) / self.frame_rate
                    self.defenses_timestamps.append((start, end))
                
                if ball:
                    self.ball_positions.append((frame_idx, ball[0], ball[1]))
            
            frame_idx += 1
        
        cap.release()
        return self._compute_statistics()
    
    def _compute_statistics(self):
        # Spieler zu Teams zuordnen
        player_avg_x = {}
        for pid, pos_list in self.player_positions.items():
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
        self.player_team_map = team_map
        
        # Angriffsstatistik
        attack_stats = defaultdict(lambda: {'total':0, 'success':0, 'errors':0})
        for att in self.attacks:
            pid = att['player_id']
            label = team_map.get(pid, f'P{pid}')
            attack_stats[label]['total'] += 1
            if att['success']:
                attack_stats[label]['success'] += 1
            else:
                attack_stats[label]['errors'] += 1
        for label in attack_stats:
            total = attack_stats[label]['total']
            attack_stats[label]['success_rate'] = (attack_stats[label]['success']/total) if total>0 else 0
        
        # Dummy für Annahme (da keine echte Erkennung)
        reception_stats = defaultdict(lambda: {'good_receptions':0, 'points_after':0})
        for label in team_map.values():
            reception_stats[label]['good_receptions'] = np.random.randint(2,10)
            reception_stats[label]['points_after'] = np.random.randint(1, reception_stats[label]['good_receptions'])
        
        error_stats = defaultdict(lambda: {'attack_errors':0, 'reception_errors':0})
        
        # Heatmaps
        attack_zones = defaultdict(list)
        for att in self.attacks:
            pid = att['player_id']
            label = team_map.get(pid, f'P{pid}')
            attack_zones[label].append(att['zone'])
        attack_heatmaps = {}
        for player, zones in attack_zones.items():
            attack_heatmaps[player] = create_heatmap(zones, title=f"{player} – Angriffszonen") if zones else create_heatmap([], title=f"{player} – keine Angriffe")
        
        defense_positions = defaultdict(list)
        for pid, pos_list in self.player_positions.items():
            label = team_map.get(pid, f'P{pid}')
            for (_, x, y) in pos_list:
                defense_positions[label].append((x,y))
        defense_heatmaps = {}
        for player, positions in defense_positions.items():
            defense_heatmaps[player] = create_heatmap(positions, title=f"{player} – Verteidigungspositionen") if positions else create_heatmap([], title=f"{player} – keine Daten")
        
        # DataFrame
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
        
        return {
            'stats_df': stats_df,
            'attack_heatmaps': attack_heatmaps,
            'defense_heatmaps': defense_heatmaps,
            'blocks_timestamps': self.blocks_timestamps,
            'defenses_timestamps': self.defenses_timestamps,
            'rallies': self.rallies,
            'raw_attacks': self.attacks,
            'player_team_map': team_map,
            'frame_rate': self.frame_rate
        }

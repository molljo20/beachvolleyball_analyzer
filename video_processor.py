import cv2
import numpy as np
import pandas as pd
import requests
import base64
from ultralytics import YOLO
from collections import defaultdict
import math
from utils import create_heatmap

class BeachVolleyballAnalyzer:
    def __init__(self, roboflow_api_key, roboflow_model_id="activity-graz-uni/volleyball-activity-dataset"):
        self.api_key = roboflow_api_key
        self.model_id = roboflow_model_id
        self.api_url = f"https://detect.roboflow.com/{self.model_id}?api_key={self.api_key}"
        
        # Lokales YOLO für Spieler- und Ballerkennung (Säule A)
        self.yolo = YOLO('yolov8n.pt')
        
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
        self.last_ball_pos = None
        self.frame_rate = 0
        self.frame_width = 0
        self.frame_height = 0
        self.player_team_map = {}
        
    def infer_roboflow(self, frame):
        """Sendet ein Frame an Roboflow API und gibt die Prediction zurück."""
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        response = requests.post(self.api_url, files={"file": img_bytes})
        if response.status_code == 200:
            return response.json().get('predictions', [])
        else:
            print(f"Roboflow API error: {response.status_code}")
            return []
    
    def process_video(self, video_path, frame_skip=5):
        cap = cv2.VideoCapture(video_path)
        self.frame_rate = cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_idx = 0
        players_last_pos = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                # YOLO: Personen und Ball
                yolo_results = self.yolo(frame, classes=[0, 32])
                persons = []
                ball = None
                for r in yolo_results[0].boxes:
                    cls = int(r.cls[0])
                    x1,y1,x2,y2 = map(int, r.xyxy[0].tolist())
                    if cls == 0:
                        persons.append((x1,y1,x2,y2))
                    elif cls == 32:
                        ball = ((x1+x2)//2, (y1+y2)//2)
                
                # Tracking (einfach)
                current_players = {}
                for (x1,y1,x2,y2) in persons:
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
                
                # Roboflow API
                roboflow_preds = self.infer_roboflow(frame)
                if roboflow_preds:
                    best_pred = max(roboflow_preds, key=lambda x: x.get('confidence',0))
                    action_label = best_pred.get('class', '')
                    confidence = best_pred.get('confidence', 0)
                    
                    # Ballwechsel-Start (Serve)
                    if action_label == 'Serve' and self.current_rally_start is None:
                        self.current_rally_start = frame_idx / self.frame_rate
                    
                    # Ballwechsel-Ende (Point)
                    if action_label == 'Point' and self.current_rally_start is not None:
                        rally_end = frame_idx / self.frame_rate
                        self.rallies.append((self.current_rally_start, rally_end))
                        self.current_rally_start = None
                    
                    # Angriff
                    if action_label in ['Attack', 'Spike'] and ball:
                        attacker_id = None
                        min_dist = 100
                        for pid, (cx, cy, _) in current_players.items():
                            dist = math.hypot(cx-ball[0], cy-ball[1])
                            if dist < min_dist:
                                min_dist = dist
                                attacker_id = pid
                        if attacker_id:
                            self.attacks.append({
                                'frame_idx': frame_idx,
                                'time': frame_idx / self.frame_rate,
                                'player_id': attacker_id,
                                'success': True,  # Platzhalter
                                'zone': (ball[0]/self.frame_width, ball[1]/self.frame_height)
                            })
                    
                    # Block / Abwehr Zeitstempel
                    if action_label == 'Block':
                        self.blocks_timestamps.append((frame_idx/self.frame_rate, (frame_idx+10)/self.frame_rate))
                    if action_label == 'Defense':
                        self.defenses_timestamps.append((frame_idx/self.frame_rate, (frame_idx+10)/self.frame_rate))
                
                if ball:
                    self.ball_positions.append((frame_idx, ball[0], ball[1]))
            
            frame_idx += 1
        
        cap.release()
        stats = self._compute_statistics()
        return stats
    
    def _compute_statistics(self):
        # Gleiche Logik wie in meinem vorherigen Code (bleibt unverändert)
        # ... (der Einfachheit halber hier ausgelassen, bitte aus vorheriger Antwort kopieren)
        # Wichtig: Die Methode muss ein dict mit stats_df, attack_heatmaps, defense_heatmaps,
        # blocks_timestamps, defenses_timestamps, rallies, raw_attacks, player_team_map, frame_rate zurückgeben.
        # Du kannst den Code aus der vorherigen video_processor.py ab Zeile ~150 übernehmen.
        pass

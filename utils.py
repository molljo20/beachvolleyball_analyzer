import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tempfile
import subprocess
import os

def cut_video_to_rallies(input_path, rallies, output_path):
    """
    Schneidet aus dem Originalvideo nur die Ballwechsel (rallies) heraus.
    rallies: Liste von (start_sec, end_sec)
    output_path: Pfad für das gekürzte Video
    """
    if not rallies:
        return None
    # Erstelle eine temporäre Datei mit der concat-Liste
    concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    for start, end in rallies:
        duration = end - start
        concat_file.write(f"file '{input_path}'\n")
        concat_file.write(f"inpoint {start}\n")
        concat_file.write(f"outpoint {end}\n")
    concat_file.close()
    
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_file.name, '-c', 'copy', output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.unlink(concat_file.name)
    return output_path

def create_heatmap(positions, field_width=1000, field_height=500, title="Heatmap"):
    """
    Erstellt eine statische Heatmap aus einer Liste von (x,y) Positionen.
    Das Spielfeld wird als Rechteck gezeichnet.
    """
    if not positions:
        # Leere Heatmap
        fig, ax = plt.subplots(figsize=(8,4))
        ax.add_patch(Rectangle((0,0), field_width, field_height, fill=False, edgecolor='black'))
        ax.set_xlim(0, field_width)
        ax.set_ylim(0, field_height)
        ax.set_title(title)
        ax.set_aspect('equal')
        return fig
    
    x_vals = [p[0] for p in positions]
    y_vals = [p[1] for p in positions]
    
    fig, ax = plt.subplots(figsize=(8,4))
    # 2D Histogramm
    hb = ax.hexbin(x_vals, y_vals, gridsize=20, cmap='hot', alpha=0.8)
    ax.add_patch(Rectangle((0,0), field_width, field_height, fill=False, edgecolor='black', linewidth=2))
    ax.set_xlim(0, field_width)
    ax.set_ylim(0, field_height)
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.colorbar(hb, ax=ax, label='Aufenthaltshäufigkeit')
    return fig

def save_fig_to_bytes(fig):
    """Speichert eine matplotlib-Figur als Bytes für PDF-Export"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

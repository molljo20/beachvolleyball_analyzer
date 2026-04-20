from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import io
from utils import save_fig_to_bytes

def export_full_pdf(results):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Titel
    story.append(Paragraph("Beachvolleyball Analysebericht", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Statistik-Tabelle
    df = results['stats_df']
    data = [df.columns.tolist()] + df.values.tolist()
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
    story.append(Paragraph("Angriffs-Heatmaps", styles['Heading2']))
    for player, fig in results['attack_heatmaps'].items():
        img_bytes = save_fig_to_bytes(fig)
        img = Image(img_bytes, width=400, height=200)
        story.append(Paragraph(f"Spieler {player}", styles['Normal']))
        story.append(img)
        story.append(Spacer(1, 10))
    
    story.append(Paragraph("Verteidigungs-Heatmaps", styles['Heading2']))
    for player, fig in results['defense_heatmaps'].items():
        img_bytes = save_fig_to_bytes(fig)
        img = Image(img_bytes, width=400, height=200)
        story.append(Paragraph(f"Spieler {player}", styles['Normal']))
        story.append(img)
        story.append(Spacer(1, 10))
    
    # Zeitstempel
    story.append(Paragraph("Erfolgreiche Blocks", styles['Heading2']))
    for ts in results['blocks_timestamps']:
        story.append(Paragraph(f"{ts[0]:.1f}s – {ts[1]:.1f}s", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Erfolgreiche Abwehraktionen", styles['Heading2']))
    for ts in results['defenses_timestamps']:
        story.append(Paragraph(f"{ts[0]:.1f}s – {ts[1]:.1f}s", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

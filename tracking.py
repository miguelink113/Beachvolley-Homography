# =========================================================================
# PIPELINE COMPLETO: TRACKING DE JUGADORES DE VÓLEY PLAYA
# =========================================================================

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import csv
import json

# =========================================================================
# 1. CONFIGURACIÓN
# =========================================================================

VIDEO_PATH = "C:\\Users\\gorka\\Documents\\Univ2526\\VisionPorComputador\\Beachvolley-Homography\\VideosAnalisis\\clip 1 ‐ Hecho con Clipchamp.mp4"
MAPA_PATH = "beachvolleyballcourt.png"
MODEL_PATH = "yolo11n.pt"
MARGIN_PERCENT = 0.10
EXPECTED_PLAYERS = 4

print("✓ Configuración cargada")


# =========================================================================
# 2. FUNCIONES AUXILIARES
# =========================================================================

def get_points(event, x, y, flags, params):
    """Callback para seleccionar puntos con el mouse."""
    points = params["points"]
    image = params["image"]
    wname = params["wname"]
    max_points = params["max_points"]

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < max_points:
        points.append([x, y])
        cv2.circle(image, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(image, str(len(points)), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imshow(wname, image)

        if len(points) == max_points:
            cv2.waitKey(300)
            cv2.destroyWindow(wname)


def point_in_polygon_with_margin(point, polygon, margin_percent=0.10):
    """Verifica si un punto está dentro de un polígono expandido."""
    center = np.mean(polygon, axis=0)
    expanded_polygon = []
    max_y = np.max(polygon[:, 1])
    
    for pt in polygon:
        if abs(pt[1] - max_y) < 5:
            direction = pt - center
            direction[1] = min(0, direction[1])
            expanded_pt = pt + direction * margin_percent
        else:
            direction = pt - center
            expanded_pt = pt + direction * margin_percent
        expanded_polygon.append(expanded_pt)
    
    expanded_polygon = np.array(expanded_polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(expanded_polygon, point, False)
    return result >= 0


def calculate_iou(box1, box2):
    """Calcula Intersection over Union entre dos bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def get_track_info(tracking_data, track_id):
    """Obtiene información completa de un track."""
    frames = []
    positions = []
    bboxes = []
    
    for frame_idx, detections in tracking_data.items():
        for det in detections:
            if det[0] == track_id:
                frames.append(frame_idx)
                positions.append((det[5], det[6]))
                bboxes.append((det[1], det[2], det[3], det[4]))
    
    if not frames:
        return None, None, [], []
    
    return min(frames), max(frames), positions, bboxes


def should_merge_ids(tracking_data, id1, id2, max_gap=5, max_distance=150, min_iou=0.3):
    """Determina si dos IDs deberían fusionarse."""
    first1, last1, positions1, bboxes1 = get_track_info(tracking_data, id1)
    first2, last2, positions2, bboxes2 = get_track_info(tracking_data, id2)
    
    if first1 is None or first2 is None:
        return False
    
    # CASO 1: Secuencial
    if last1 < first2:
        gap = first2 - last1
        if gap <= max_gap:
            last_pos1 = positions1[-1]
            first_pos2 = positions2[0]
            distance = np.sqrt((last_pos1[0] - first_pos2[0])**2 + 
                              (last_pos1[1] - first_pos2[1])**2)
            
            last_bbox1 = bboxes1[-1]
            first_bbox2 = bboxes2[0]
            size1 = (last_bbox1[2] - last_bbox1[0]) * (last_bbox1[3] - last_bbox1[1])
            size2 = (first_bbox2[2] - first_bbox2[0]) * (first_bbox2[3] - first_bbox2[1])
            size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0
            
            if distance <= max_distance and size_ratio > 0.5:
                return True
    
    # CASO 2: Solapamiento
    overlap_start = max(first1, first2)
    overlap_end = min(last1, last2)
    
    if overlap_start <= overlap_end:
        overlapping_frames = []
        for frame_idx in range(overlap_start, overlap_end + 1):
            if frame_idx not in tracking_data:
                continue
            
            bbox1, bbox2 = None, None
            pos1, pos2 = None, None
            
            for det in tracking_data[frame_idx]:
                if det[0] == id1:
                    bbox1 = (det[1], det[2], det[3], det[4])
                    pos1 = (det[5], det[6])
                if det[0] == id2:
                    bbox2 = (det[1], det[2], det[3], det[4])
                    pos2 = (det[5], det[6])
            
            if bbox1 and bbox2:
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                iou = calculate_iou(bbox1, bbox2)
                overlapping_frames.append((distance, iou))
        
        if overlapping_frames:
            avg_distance = np.mean([d for d, _ in overlapping_frames])
            avg_iou = np.mean([iou for _, iou in overlapping_frames])
            
            if avg_iou >= min_iou or avg_distance <= max_distance * 0.5:
                return True
    
    return False


def merge_track_ids(tracking_data, id_from, id_to):
    """Fusiona id_from en id_to."""
    for frame_idx in tracking_data:
        new_detections = []
        for det in tracking_data[frame_idx]:
            if det[0] == id_from:
                new_det = (id_to,) + det[1:]
                new_detections.append(new_det)
            else:
                new_detections.append(det)
        tracking_data[frame_idx] = new_detections


def count_frames_with_excess(tracking_data, max_expected=4):
    """Cuenta frames con más jugadores del esperado."""
    return sum(1 for dets in tracking_data.values() if len(dets) > max_expected)


print("✓ Funciones auxiliares definidas")


# =========================================================================
# 3. CARGAR VIDEO Y MAPA
# =========================================================================

video = cv2.VideoCapture(VIDEO_PATH)
if not video.isOpened():
    raise RuntimeError(f"No se pudo abrir el video: {VIDEO_PATH}")

fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✓ Video cargado: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")

ret, first_frame = video.read()
if not ret:
    raise RuntimeError("No se pudo leer el primer frame")

mapa = cv2.imread(MAPA_PATH)
if mapa is None:
    raise FileNotFoundError(f"No se pudo cargar el mapa: {MAPA_PATH}")

print(f"✓ Mapa cargado: {mapa.shape[1]}x{mapa.shape[0]}")


# =========================================================================
# 4. SELECCIÓN DE PUNTOS DEL CAMPO
# =========================================================================

puntos_campo = []
N = 4

imgA = first_frame.copy()
cv2.namedWindow("Selecciona 4 esquinas del campo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Selecciona 4 esquinas del campo", 1200, 800)
cv2.imshow("Selecciona 4 esquinas del campo", imgA)

cv2.setMouseCallback(
    "Selecciona 4 esquinas del campo",
    get_points,
    {"points": puntos_campo, "image": imgA, 
     "wname": "Selecciona 4 esquinas del campo", "max_points": N}
)

print("Marca las 4 esquinas del campo (arriba-izq, arriba-der, abajo-der, abajo-izq)")
cv2.waitKey(0)
cv2.destroyAllWindows()

puntos_campo = np.array(puntos_campo, dtype=np.float32)
print(f"✓ {len(puntos_campo)} puntos seleccionados")


# =========================================================================
# 5. CARGAR MODELO YOLO
# =========================================================================

model = YOLO(MODEL_PATH)
print("✓ Modelo YOLO cargado")


# =========================================================================
# 6. TRACKING OFFLINE
# =========================================================================

video.set(cv2.CAP_PROP_POS_FRAMES, 0)
tracking_data = {}

print(f"\nProcesando {total_frames} frames...")
print("Esto puede tardar unos minutos...\n")

frame_idx = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    results = model.track(frame, persist=True, verbose=False, classes=[0])
    frame_detections = []
    
    for r in results:
        if r.boxes.id is None:
            continue
            
        for box, track_id in zip(r.boxes, r.boxes.id):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = y2
            
            if point_in_polygon_with_margin((cx, cy), puntos_campo, MARGIN_PERCENT):
                frame_detections.append((
                    int(track_id.item()),
                    x1, y1, x2, y2,
                    cx, cy
                ))
    
    tracking_data[frame_idx] = frame_detections
    
    if frame_idx % 50 == 0:
        progress = (frame_idx / total_frames) * 100
        print(f"  Frame {frame_idx}/{total_frames} ({progress:.1f}%) - {len(frame_detections)} jugadores detectados")
    
    frame_idx += 1

print(f"\n✓ Tracking completado: {len(tracking_data)} frames procesados")

# Estadísticas iniciales
all_track_ids = set()
for dets in tracking_data.values():
    for det in dets:
        all_track_ids.add(det[0])

total_detections = sum(len(dets) for dets in tracking_data.values())
print(f"  Total detecciones: {total_detections}")
print(f"  IDs únicos detectados: {len(all_track_ids)}")
print(f"  IDs: {sorted(all_track_ids)}")


# =========================================================================
# 7. CORRECCIÓN AVANZADA DE IDS DUPLICADOS
# =========================================================================

print("\n" + "=" * 70)
print("CORRECCIÓN AVANZADA DE IDS DUPLICADOS")
print("=" * 70)

all_ids = sorted(all_track_ids)
print(f"\n📊 Estado inicial:")
print(f"   IDs detectados: {all_ids}")
print(f"   Total IDs: {len(all_ids)}")
print(f"   Frames con >4 jugadores: {count_frames_with_excess(tracking_data)}")

merge_map = {id: id for id in all_ids}
total_merges = 0

params = [
    (5, 100, 0.4, "Muy estricto - gaps pequeños"),
    (10, 150, 0.3, "Estricto - gaps medianos"),
    (15, 200, 0.25, "Moderado - gaps más largos"),
    (20, 250, 0.2, "Permisivo - oclusiones largas"),
    (30, 350, 0.15, "Muy permisivo - último intento"),
]

for iteration, (max_gap, max_distance, min_iou, description) in enumerate(params, 1):
    print(f"\n{'─' * 70}")
    print(f"ITERACIÓN {iteration}: {description}")
    print(f"   Parámetros: gap≤{max_gap}f, dist≤{max_distance}px, IoU≥{min_iou}")
    print(f"{'─' * 70}")
    
    current_ids = set()
    for dets in tracking_data.values():
        for det in dets:
            current_ids.add(det[0])
    current_ids = sorted(current_ids)
    
    frames_excess = count_frames_with_excess(tracking_data)
    
    print(f"   IDs actuales: {current_ids} ({len(current_ids)} IDs)")
    print(f"   Frames problemáticos: {frames_excess}")
    
    if len(current_ids) <= EXPECTED_PLAYERS and frames_excess < 10:
        print(f"   ✅ ¡Objetivo alcanzado!")
        break
    
    merge_candidates = []
    
    for i, id1 in enumerate(current_ids):
        for id2 in current_ids[i+1:]:
            if should_merge_ids(tracking_data, id1, id2, max_gap, max_distance, min_iou):
                impact = 0
                for frame_idx, dets in tracking_data.items():
                    ids_in_frame = [d[0] for d in dets]
                    if id1 in ids_in_frame and id2 in ids_in_frame:
                        impact += 1
                
                merge_candidates.append((id1, id2, impact))
    
    merge_candidates.sort(key=lambda x: x[2], reverse=True)
    print(f"   Candidatos encontrados: {len(merge_candidates)}")
    
    if not merge_candidates:
        print(f"   ⚠️  No se encontraron fusiones posibles")
        continue
    
    merges_in_iteration = 0
    for id1, id2, impact in merge_candidates:
        current_id1 = merge_map.get(id1, id1)
        current_id2 = merge_map.get(id2, id2)
        
        if current_id1 == current_id2:
            continue
        
        merge_from = max(current_id1, current_id2)
        merge_to = min(current_id1, current_id2)
        
        print(f"      → Fusionando ID {merge_from} → ID {merge_to} (resuelve {impact} frames)")
        
        merge_track_ids(tracking_data, merge_from, merge_to)
        
        for key in merge_map:
            if merge_map[key] == merge_from:
                merge_map[key] = merge_to
        merge_map[merge_from] = merge_to
        
        merges_in_iteration += 1
        total_merges += 1
    
    print(f"   ✓ Fusiones realizadas: {merges_in_iteration}")

print(f"\n{'=' * 70}")
print("✅ CORRECCIÓN COMPLETADA")
print(f"{'=' * 70}")

final_ids = set()
for dets in tracking_data.values():
    for det in dets:
        final_ids.add(det[0])
final_ids = sorted(final_ids)

print(f"\n📊 Resumen de cambios:")
print(f"   IDs originales: {len(all_ids)} → IDs finales: {len(final_ids)}")
print(f"   Total de fusiones: {total_merges}")
print(f"   IDs finales: {final_ids}")

print(f"\n📈 Distribución de jugadores por frame:")
distribution = defaultdict(int)
for dets in tracking_data.values():
    distribution[len(dets)] += 1

for num_players in sorted(distribution.keys()):
    count = distribution[num_players]
    percentage = (count / len(tracking_data)) * 100
    bar = "█" * int(percentage / 2)
    marker = "✓" if num_players == EXPECTED_PLAYERS else "⚠" if num_players > EXPECTED_PLAYERS else "!"
    print(f"   {marker} {num_players} jugadores: {count:4d} frames ({percentage:5.1f}%) {bar}")

frames_exact = distribution.get(EXPECTED_PLAYERS, 0)
frames_over = sum(count for num, count in distribution.items() if num > EXPECTED_PLAYERS)
frames_under = sum(count for num, count in distribution.items() if num < EXPECTED_PLAYERS)

print(f"\n🎯 Calidad del tracking:")
print(f"   Frames perfectos (4 jugadores): {frames_exact} ({frames_exact/len(tracking_data)*100:.1f}%)")
print(f"   Frames con exceso (>4): {frames_over} ({frames_over/len(tracking_data)*100:.1f}%)")
print(f"   Frames con déficit (<4): {frames_under} ({frames_under/len(tracking_data)*100:.1f}%)")

if len(final_ids) == EXPECTED_PLAYERS:
    print(f"\n🎉 ¡PERFECTO! Exactamente {EXPECTED_PLAYERS} jugadores detectados")
elif len(final_ids) < EXPECTED_PLAYERS:
    print(f"\n⚠️  Solo {len(final_ids)} jugadores detectados (esperados: {EXPECTED_PLAYERS})")
else:
    print(f"\n⚠️  {len(final_ids)} jugadores detectados (esperados: {EXPECTED_PLAYERS})")

all_track_ids = final_ids


# =========================================================================
# 8. ANÁLISIS FINAL
# =========================================================================

print(f"\n{'=' * 70}")
print("ANÁLISIS FINAL DEL TRACKING")
print(f"{'=' * 70}")

track_durations_final = defaultdict(int)

for frame_idx, detections in tracking_data.items():
    for det in detections:
        track_id = det[0]
        track_durations_final[track_id] += 1

print("\nDuración de cada track (en frames):")
for track_id in sorted(track_durations_final.keys()):
    duration = track_durations_final[track_id]
    duration_sec = duration / fps
    percentage = (duration / total_frames) * 100
    print(f"  ID {track_id}: {duration} frames ({duration_sec:.1f}s, {percentage:.1f}% del video)")

print(f"\n📊 Resumen:")
print(f"  • Jugadores únicos: {len(final_ids)}")
print(f"  • Frame más poblado: {max(len(dets) for dets in tracking_data.values())} jugadores")
print(f"  • Frame menos poblado: {min(len(dets) for dets in tracking_data.values())} jugadores")


# =========================================================================
# 9. EXPORTAR DATOS A CSV
# =========================================================================

print(f"\n{'=' * 70}")
print("EXPORTANDO DATOS")
print(f"{'=' * 70}\n")

tracking_list = []

for frame_idx, detections in sorted(tracking_data.items()):
    for det in detections:
        track_id, x1, y1, x2, y2, cx, cy = det
        tracking_list.append({
            'frame': frame_idx,
            'track_id': track_id,
            'bbox_x1': x1,
            'bbox_y1': y1,
            'bbox_x2': x2,
            'bbox_y2': y2,
            'center_x': cx,
            'center_y': cy,
            'timestamp_sec': frame_idx / fps
        })

csv_filename = "tracking_data.csv"
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['frame', 'track_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 
                  'center_x', 'center_y', 'timestamp_sec']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(tracking_list)

print(f"✓ Datos exportados a '{csv_filename}'")
print(f"  Total de registros: {len(tracking_list)}")

summary_data = {
    'video_info': {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height
    },
    'campo_puntos': puntos_campo.tolist(),
    'jugadores_ids': sorted(final_ids),
    'num_jugadores': len(final_ids),
    'total_detecciones': len(tracking_list)
}

json_filename = "tracking_summary.json"
with open(json_filename, 'w', encoding='utf-8') as jsonfile:
    json.dump(summary_data, jsonfile, indent=2)

print(f"✓ Resumen exportado a '{json_filename}'")

for track_id in sorted(final_ids):
    player_data = [d for d in tracking_list if d['track_id'] == track_id]
    player_filename = f"tracking_player_{track_id}.csv"
    
    with open(player_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(player_data)
    
    print(f"  - '{player_filename}': {len(player_data)} posiciones")

print(f"\n📁 Archivos generados en: {os.getcwd()}")


# =========================================================================
# 10. VISUALIZACIÓN DEL RESULTADO
# =========================================================================

print(f"\n{'=' * 70}")
print("VISUALIZACIÓN DEL TRACKING")
print(f"{'=' * 70}\n")

video.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0

center = np.mean(puntos_campo, axis=0)
max_y = np.max(puntos_campo[:, 1])
expanded_polygon = []

for pt in puntos_campo:
    if abs(pt[1] - max_y) < 5:
        direction = pt - center
        direction[1] = min(0, direction[1])
        expanded_pt = pt + direction * MARGIN_PERCENT
    else:
        direction = pt - center
        expanded_pt = pt + direction * MARGIN_PERCENT
    expanded_polygon.append(expanded_pt)

expanded_polygon = np.array(expanded_polygon, dtype=np.int32)

print("Reproduciendo video con tracking corregido...")
print("Presiona ESC para salir\n")

cv2.namedWindow("Tracking Offline - Jugadores", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking Offline - Jugadores", 1200, 800)

while True:
    ret, frame = video.read()
    if not ret:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        continue
    
    cv2.polylines(frame, [expanded_polygon], True, (0, 255, 255), 2)
    
    if frame_idx in tracking_data:
        detections = tracking_data[frame_idx]
        
        for det in detections:
            track_id, x1, y1, x2, y2, cx, cy = det
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    info_text = f"Frame: {frame_idx}/{total_frames} | Jugadores: {len(tracking_data.get(frame_idx, []))}"
    cv2.putText(frame, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow("Tracking Offline - Jugadores", frame)
    
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break
    
    frame_idx += 1

video.release()
cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("✅ PIPELINE COMPLETADO")
print("=" * 70)
print(f"\n📊 Resumen final:")
print(f"   • Video procesado: {total_frames} frames")
print(f"   • Jugadores detectados: {len(final_ids)}")
print(f"   • Detecciones totales: {len(tracking_list)}")
print(f"   • Fusiones realizadas: {total_merges}")
print(f"   • Calidad: {frames_exact/len(tracking_data)*100:.1f}% frames perfectos")
print(f"\n✓ Visualización finalizada")
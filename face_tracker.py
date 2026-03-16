#!/usr/bin/env python3
"""
face_tracker.py
===============
Analiza un video y genera un crop_map.json con la posición X del rostro
por cada segundo, suavizada con interpolación para crop dinámico fluido.

Uso:
    python face_tracker.py \
        --video  /mnt/nas/predicas/20260315_wxM8MkMKvNE.mp4 \
        --output ./output/shorts/crop_map.json \
        [--sample-rate 1]     # frames a analizar por segundo (default: 1)
        [--smooth-window 3]   # segundos de ventana de suavizado (default: 3)
        [--debug]

Salida (crop_map.json):
    {
        "video":        "20260315_wxM8MkMKvNE.mp4",
        "src_width":    1920,
        "src_height":   1080,
        "crop_width":   607,
        "sample_rate":  1,
        "smooth_window": 3,
        "entries": [
            {"t": 0.0,  "crop_x": 612, "face_detected": true},
            {"t": 1.0,  "crop_x": 618, "face_detected": true},
            {"t": 2.0,  "crop_x": 612, "face_detected": false},  ← interpolado
            ...
        ]
    }

Dependencias:
    pip install mediapipe opencv-python-headless

Notas:
    - opencv-python-headless es preferible en servidores sin display
    - MediaPipe funciona en CPU sin GPU ni Coral
    - El Coral TPU puede usarse en el futuro con tflite delegate
"""

import argparse
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("face_tracker")


# ──────────────────────────────────────────────
# IMPORTS CON MENSAJES DE ERROR CLAROS
# ──────────────────────────────────────────────

def import_deps():
    """Importa cv2 y mediapipe con mensajes de instalación claros."""
    try:
        import cv2
    except ImportError:
        log.error(
            "opencv no instalado.\n"
            "Instalalo con:  pip install opencv-python-headless"
        )
        sys.exit(1)

    try:
        import mediapipe as mp
    except ImportError:
        log.error(
            "mediapipe no instalado.\n"
            "Instalalo con:  pip install mediapipe"
        )
        sys.exit(1)

    return cv2, mp


# ──────────────────────────────────────────────
# VIDEO INFO
# ──────────────────────────────────────────────

def get_video_info(video_path: Path) -> dict:
    """Obtiene fps, width, height, duration vía ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("ffprobe falló: %s", r.stderr)
        sys.exit(1)

    data     = json.loads(r.stdout)
    stream   = data["streams"][0]
    fmt      = data["format"]

    # r_frame_rate viene como "30000/1001" → calcular
    num, den = stream["r_frame_rate"].split("/")
    fps      = float(num) / float(den)
    duration = float(fmt["duration"])

    return {
        "width":    int(stream["width"]),
        "height":   int(stream["height"]),
        "fps":      fps,
        "duration": duration,
    }


# ──────────────────────────────────────────────
# CROP WIDTH (igual que generate_shorts.py)
# ──────────────────────────────────────────────

def compute_crop_width(src_w: int, src_h: int) -> int:
    """Calcula el ancho del crop 9:16 para el video fuente."""
    crop_w = math.floor(src_h * 9 / 16)
    if crop_w > src_w:
        crop_w = src_w
    return crop_w


# ──────────────────────────────────────────────
# FACE DETECTION
# ──────────────────────────────────────────────

def detect_face_center_x(frame, face_detection, src_w: int) -> int | None:
    """
    Detecta el rostro más prominente en un frame y retorna
    el centro X en píxeles del frame original.
    Retorna None si no se detecta ningún rostro.
    """
    import mediapipe as mp

    # MediaPipe espera RGB
    rgb = frame[:, :, ::-1]
    results = face_detection.process(rgb)

    if not results.detections:
        return None

    # Tomar la detección con mayor score
    best = max(results.detections, key=lambda d: d.score[0])
    bbox = best.location_data.relative_bounding_box

    # Centro X de la bounding box en píxeles
    center_x_rel = bbox.xmin + bbox.width / 2
    center_x_px  = int(center_x_rel * src_w)

    return center_x_px


# ──────────────────────────────────────────────
# CROP_X desde center_x
# ──────────────────────────────────────────────

def center_to_crop_x(center_x: int, crop_w: int, src_w: int) -> int:
    """
    Convierte el centro X del rostro a crop_x (esquina izquierda del crop).
    Clampea para que el crop no se salga del frame.
    """
    crop_x = center_x - crop_w // 2
    crop_x = max(0, min(crop_x, src_w - crop_w))
    return crop_x


# ──────────────────────────────────────────────
# SUAVIZADO CON VENTANA DESLIZANTE
# ──────────────────────────────────────────────

def smooth_crop_positions(entries: list[dict], window_sec: int) -> list[dict]:
    """
    Suaviza la posición crop_x usando una ventana deslizante centrada.
    Solo suaviza entradas con face_detected=True.
    Las entradas sin detección se interpolan linealmente entre vecinos detectados.

    Proceso:
      1. Rellenar gaps (face_detected=False) por interpolación lineal
      2. Suavizar toda la secuencia con ventana promedio
    """
    n = len(entries)
    if n == 0:
        return entries

    # ── Paso 1: Rellenar gaps por interpolación lineal ──
    # Construir lista de índices con detección válida
    detected_indices = [i for i, e in enumerate(entries) if e["face_detected"]]

    if not detected_indices:
        log.warning("Ningún frame con rostro detectado — usando crop centrado")
        return entries

    # Para cada gap, interpolar entre el vecino anterior y siguiente detectado
    filled = [e["crop_x"] for e in entries]

    for i in range(n):
        if not entries[i]["face_detected"]:
            # Buscar vecinos más cercanos con detección
            prev_idx = next((j for j in range(i - 1, -1, -1) if entries[j]["face_detected"]), None)
            next_idx = next((j for j in range(i + 1, n)      if entries[j]["face_detected"]), None)

            if prev_idx is not None and next_idx is not None:
                # Interpolación lineal
                t = (i - prev_idx) / (next_idx - prev_idx)
                filled[i] = int(filled[prev_idx] + t * (filled[next_idx] - filled[prev_idx]))
            elif prev_idx is not None:
                filled[i] = filled[prev_idx]
            elif next_idx is not None:
                filled[i] = filled[next_idx]

    # ── Paso 2: Suavizado con ventana deslizante ──
    half = window_sec // 2
    smoothed = []

    for i in range(n):
        lo  = max(0, i - half)
        hi  = min(n, i + half + 1)
        avg = int(sum(filled[lo:hi]) / (hi - lo))
        smoothed.append(avg)

    # Actualizar entries
    result = []
    for i, entry in enumerate(entries):
        result.append({
            "t":              entry["t"],
            "crop_x":         smoothed[i],
            "face_detected":  entry["face_detected"],
        })

    return result


# ──────────────────────────────────────────────
# ANÁLISIS PRINCIPAL
# ──────────────────────────────────────────────

def analyze_video(
    video_path:   Path,
    sample_rate:  int,
    smooth_window: int,
) -> dict:
    """
    Analiza el video frame a frame (a sample_rate fps),
    detecta rostros y genera el crop_map.
    """
    cv2, mp = import_deps()

    info    = get_video_info(video_path)
    src_w   = info["width"]
    src_h   = info["height"]
    fps     = info["fps"]
    dur     = info["duration"]
    crop_w  = compute_crop_width(src_w, src_h)

    # Intervalo en frames entre muestras
    frame_interval = max(1, int(fps / sample_rate))
    total_samples  = int(dur * sample_rate)

    log.info("Video: %dx%d  %.1fs  %.1ffps", src_w, src_h, dur, fps)
    log.info("Crop width 9:16: %dpx", crop_w)
    log.info("Sample rate: %d fps → %d muestras totales", sample_rate, total_samples)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("No se pudo abrir el video: %s", video_path)
        sys.exit(1)

    mp_face = mp.solutions.face_detection

    entries        = []
    frame_idx      = 0
    sample_idx     = 0
    detected_count = 0

    with mp_face.FaceDetection(
        model_selection=1,       # 1 = modelo para rostros lejanos (mejor para prédicas)
        min_detection_confidence=0.5,
    ) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Solo procesar frames en el intervalo definido
            if frame_idx % frame_interval == 0:
                t_sec      = frame_idx / fps
                center_x   = detect_face_center_x(frame, face_detection, src_w)
                detected   = center_x is not None

                if detected:
                    crop_x = center_to_crop_x(center_x, crop_w, src_w)
                    detected_count += 1
                else:
                    # Placeholder — se rellenará en smooth_crop_positions
                    crop_x = (src_w - crop_w) // 2  # centro por defecto

                entries.append({
                    "t":             round(t_sec, 3),
                    "crop_x":        crop_x,
                    "face_detected": detected,
                })

                sample_idx += 1

                # Progreso cada 10%
                if total_samples > 0 and sample_idx % max(1, total_samples // 10) == 0:
                    pct = sample_idx / total_samples * 100
                    log.info(
                        "  Progreso: %d%% (%d/%d muestras, %d rostros detectados)",
                        pct, sample_idx, total_samples, detected_count
                    )

            frame_idx += 1

    cap.release()

    detection_rate = detected_count / max(1, len(entries)) * 100
    log.info(
        "Análisis completo: %d muestras, %.0f%% con rostro detectado",
        len(entries), detection_rate
    )

    if detection_rate < 30:
        log.warning(
            "Tasa de detección baja (%.0f%%). "
            "El video puede tener encuadres amplios o iluminación difícil. "
            "Considera ajustar --sample-rate o revisar el video.",
            detection_rate
        )

    # Suavizar
    log.info("Aplicando suavizado (ventana %ds)...", smooth_window)
    smoothed_entries = smooth_crop_positions(entries, smooth_window)

    return {
        "video":         video_path.name,
        "src_width":     src_w,
        "src_height":    src_h,
        "crop_width":    crop_w,
        "sample_rate":   sample_rate,
        "smooth_window": smooth_window,
        "duration":      round(dur, 3),
        "detection_rate_pct": round(detection_rate, 1),
        "entries":       smoothed_entries,
    }


# ──────────────────────────────────────────────
# QUERY: obtener crop_x para un tiempo dado
# ──────────────────────────────────────────────

def get_crop_x_at(crop_map: dict, t_sec: float) -> int:
    """
    Retorna el crop_x interpolado para un tiempo t_sec dado.
    Usado por generate_shorts.py.
    """
    entries = crop_map["entries"]
    if not entries:
        # Fallback: centro
        return (crop_map["src_width"] - crop_map["crop_width"]) // 2

    # Búsqueda binaria manual del intervalo
    lo, hi = 0, len(entries) - 1

    if t_sec <= entries[0]["t"]:
        return entries[0]["crop_x"]
    if t_sec >= entries[-1]["t"]:
        return entries[-1]["crop_x"]

    while lo < hi - 1:
        mid = (lo + hi) // 2
        if entries[mid]["t"] <= t_sec:
            lo = mid
        else:
            hi = mid

    # Interpolación lineal entre lo y hi
    t0, x0 = entries[lo]["t"], entries[lo]["crop_x"]
    t1, x1 = entries[hi]["t"], entries[hi]["crop_x"]

    if t1 == t0:
        return x0

    alpha  = (t_sec - t0) / (t1 - t0)
    result = int(x0 + alpha * (x1 - x0))
    return result


# ──────────────────────────────────────────────
# ARG PARSER
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Analiza un video y genera crop_map.json para face tracking."
    )
    p.add_argument("--video",          required=True, help="Video de entrada .mp4")
    p.add_argument("--output",         required=True, help="Ruta del crop_map.json a generar")
    p.add_argument("--sample-rate",    type=int,   default=1,
                   help="Frames a analizar por segundo (default: 1)")
    p.add_argument("--smooth-window",  type=int,   default=3,
                   help="Segundos de ventana de suavizado (default: 3)")
    p.add_argument("--debug",          action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    video_path  = Path(args.video)
    output_path = Path(args.output)

    if not video_path.exists():
        log.error("Video no encontrado: %s", video_path)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("Face tracker iniciado")
    log.info("Video:        %s", video_path)
    log.info("Sample rate:  %d fps", args.sample_rate)
    log.info("Smooth window: %ds", args.smooth_window)
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    crop_map = analyze_video(video_path, args.sample_rate, args.smooth_window)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(crop_map, f, indent=2, ensure_ascii=False)

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("crop_map.json guardado en: %s", output_path)
    log.info("Entradas generadas: %d", len(crop_map["entries"]))
    log.info("Detección: %.1f%%", crop_map["detection_rate_pct"])
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()

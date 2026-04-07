#!/usr/bin/env python3
"""
face_tracker.py  v2  —  Optical Flow tracker
=============================================
Reemplaza la detección de rostros (MediaPipe) por Lucas-Kanade Optical Flow.

Por qué optical flow para cámara fija:
  - Cámara fija = fondo estático → cualquier movimiento de píxeles es el sujeto
  - No depende de que el rostro sea visible (pastor de espaldas, de costado, etc.)
  - Sin modelos ML — solo OpenCV puro, más rápido y más estable
  - Detección de zona de movimiento dominante → centroide = posición del sujeto

Pipeline:
  1. Muestrear pares de frames (frame A, frame B separados ~0.5s)
  2. Detectar puntos buenos para trackear en frame A (Shi-Tomasi)
  3. Calcular flow de esos puntos hacia frame B (Lucas-Kanade)
  4. Filtrar puntos con movimiento significativo (descartar fondo casi estático)
  5. Centroide de los puntos que se movieron = posición X del sujeto
  6. Aplicar EMA bidireccional sobre la serie de posiciones
  7. Aplicar límite de velocidad máxima (px/seg) para pans suaves

Salida (crop_map.json) — mismo formato que v1, compatible con generate_shorts.py:
    {
        "video":              "...",
        "src_width":          1920,
        "src_height":         1080,
        "crop_width":         607,
        "tracker":            "optical_flow",
        "sample_rate":        1,
        "ema_alpha":          0.3,
        "max_pan_speed":      80,
        "duration":           2294.6,
        "detection_rate_pct": 94.1,
        "entries": [
            {"t": 0.0, "crop_x": 612, "face_detected": true},
            ...
        ]
    }

Dependencias:
    pip install opencv-python-headless

Uso:
    python face_tracker.py \
        --video  /mnt/nas/predicas/video.mp4 \
        --output ./data/output/shorts/video_crop_map.json \
        [--sample-rate   1]      frames analizados por segundo (default: 1)
        [--flow-interval 0.5]    segundos entre frame A y frame B del flow (default: 0.5)
        [--min-motion    2.0]    píxeles mínimos de movimiento para considerar un punto (default: 2.0)
        [--ema-alpha     0.15]   suavizado EMA (default: 0.15 — más suave que v1)
        [--max-pan-speed 80]     máximo px/seg que puede moverse el crop (default: 80)
        [--confidence    0.0]    no usado (compatibilidad con v1)
        [--debug]
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
# IMPORTS
# ──────────────────────────────────────────────

def import_cv2():
    try:
        import cv2
        import numpy as np
        return cv2, np
    except ImportError:
        log.error("opencv no instalado.\nInstalalo: pip install opencv-python-headless")
        sys.exit(1)


# ──────────────────────────────────────────────
# VIDEO INFO
# ──────────────────────────────────────────────

def get_video_info(video_path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
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
    num, den = stream["r_frame_rate"].split("/")
    fps      = float(num) / float(den)
    duration = float(data["format"]["duration"])

    return {
        "width":    int(stream["width"]),
        "height":   int(stream["height"]),
        "fps":      fps,
        "duration": duration,
    }


# ──────────────────────────────────────────────
# CROP HELPERS
# ──────────────────────────────────────────────

def compute_crop_width(src_w: int, src_h: int) -> int:
    return min(math.floor(src_h * 9 / 16), src_w)


def center_to_crop_x(center_x: int, crop_w: int, src_w: int) -> int:
    return max(0, min(center_x - crop_w // 2, src_w - crop_w))


# ──────────────────────────────────────────────
# OPTICAL FLOW — detección de zona de movimiento
# ──────────────────────────────────────────────

def detect_motion_center_x(
    frame_a,
    frame_b,
    min_motion_px: float,
    cv2,
    np,
) -> int | None:
    """
    Calcula el centroide X de la zona de movimiento entre frame_a y frame_b
    usando Lucas-Kanade Optical Flow.

    Retorna:
        int  — coordenada X del centroide de movimiento en píxeles
        None — si no hay movimiento significativo (escena estática)

    Por qué Shi-Tomasi + LK en lugar de Farneback dense flow:
        - Shi-Tomasi detecta esquinas/texturas (ropa, cara, manos)
        - LK trackea esos puntos específicos → mucho más rápido que dense
        - Para un sujeto en movimiento, los puntos de mayor textura
          que se desplazan son el sujeto, no el fondo liso
    """
    h, w = frame_a.shape[:2]

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    # Detectar puntos buenos para trackear en frame_a
    # maxCorners alto para cubrir bien al sujeto
    feature_params = dict(
        maxCorners  = 300,
        qualityLevel= 0.01,
        minDistance = 10,
        blockSize   = 7,
    )
    pts_a = cv2.goodFeaturesToTrack(gray_a, mask=None, **feature_params)

    if pts_a is None or len(pts_a) < 5:
        return None

    # Lucas-Kanade optical flow
    lk_params = dict(
        winSize   = (21, 21),
        maxLevel  = 3,
        criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    pts_b, status, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, pts_a, None, **lk_params)

    if pts_b is None:
        return None

    # Filtrar puntos con tracking exitoso
    good_a = pts_a[status == 1]
    good_b = pts_b[status == 1]

    if len(good_a) < 3:
        return None

    # Calcular magnitud de movimiento por punto
    displacement = good_b - good_a
    magnitude    = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2)

    # Filtrar puntos con movimiento significativo
    # (descartar puntos casi estáticos = fondo)
    moving_mask = magnitude > min_motion_px
    moving_pts  = good_b[moving_mask]

    if len(moving_pts) < 3:
        # Escena estática — nadie se mueve en este intervalo
        return None

    # Centroide X de los puntos en movimiento
    center_x = int(np.median(moving_pts[:, 0]))
    return center_x


# ──────────────────────────────────────────────
# SUAVIZADO: EMA BIDIRECCIONAL + LÍMITE DE VELOCIDAD
# ──────────────────────────────────────────────

def fill_gaps(values: list[int | None], fallback: int) -> list[int]:
    """
    Rellena None por interpolación lineal entre vecinos válidos.
    Los extremos sin vecino usan el fallback (centro del frame).
    """
    n      = len(values)
    filled = list(values)

    for i in range(n):
        if filled[i] is not None:
            continue
        prev = next((j for j in range(i - 1, -1, -1) if values[j] is not None), None)
        nxt  = next((j for j in range(i + 1, n)      if values[j] is not None), None)

        if prev is not None and nxt is not None:
            t         = (i - prev) / (nxt - prev)
            filled[i] = int(filled[prev] + t * (filled[nxt] - filled[prev]))
        elif prev is not None:
            filled[i] = filled[prev]
        elif nxt is not None:
            filled[i] = filled[nxt]
        else:
            filled[i] = fallback

    return filled


def apply_ema_bidirectional(values: list[int], alpha: float) -> list[float]:
    """
    EMA bidireccional (forward + backward promediados).
    Sin lag, sin overshooting.
    """
    n = len(values)
    if n == 0:
        return values

    fwd    = [0.0] * n
    fwd[0] = float(values[0])
    for i in range(1, n):
        fwd[i] = alpha * values[i] + (1 - alpha) * fwd[i - 1]

    bwd       = [0.0] * n
    bwd[n-1]  = float(values[-1])
    for i in range(n - 2, -1, -1):
        bwd[i] = alpha * values[i] + (1 - alpha) * bwd[i + 1]

    return [(fwd[i] + bwd[i]) / 2.0 for i in range(n)]


def apply_speed_limit(
    crop_xs:      list[float],
    timestamps:   list[float],
    max_px_per_sec: float,
    src_w:        int,
    crop_w:       int,
) -> list[int]:
    """
    Limita la velocidad máxima de movimiento del crop.
    Si entre dos keyframes el crop debería moverse más de max_px_per_sec,
    lo frena — el sujeto puede salirse momentáneamente del cuadro,
    pero el movimiento de cámara se ve deliberado y suave.

    Este es el mismo principio que usa un camarógrafo humano:
    no sigue cada micro-movimiento, hace pans lentos.
    """
    n      = len(crop_xs)
    result = [0.0] * n
    result[0] = crop_xs[0]

    for i in range(1, n):
        dt       = timestamps[i] - timestamps[i - 1]
        max_move = max_px_per_sec * dt
        delta    = crop_xs[i] - result[i - 1]
        # Clampear el movimiento al máximo permitido
        clamped  = max(-max_move, min(max_move, delta))
        result[i] = result[i - 1] + clamped

    max_x = src_w - crop_w
    return [max(0, min(int(x), max_x)) for x in result]


# ──────────────────────────────────────────────
# ANÁLISIS PRINCIPAL
# ──────────────────────────────────────────────

def analyze_video(
    video_path:      Path,
    sample_rate:     int,
    flow_interval:   float,
    min_motion_px:   float,
    ema_alpha:       float,
    max_pan_speed:   float,
) -> dict:

    cv2, np = import_cv2()

    info    = get_video_info(video_path)
    src_w   = info["width"]
    src_h   = info["height"]
    fps     = info["fps"]
    dur     = info["duration"]
    crop_w  = compute_crop_width(src_w, src_h)
    fallback_x = center_to_crop_x(src_w // 2, crop_w, src_w)  # centro del frame

    # Frames necesarios para el flow
    # frame_a en t, frame_b en t + flow_interval
    flow_frames = max(1, int(fps * flow_interval))
    # Intervalo entre muestras
    sample_frames = max(1, int(fps / sample_rate))
    total_samples = int(dur * sample_rate)

    log.info("Video: %dx%d  %.1fs  %.1f fps", src_w, src_h, dur, fps)
    log.info("Crop 9:16: %dpx ancho  |  centro fallback: x=%d", crop_w, fallback_x)
    log.info("Sample rate: %d fps  |  Flow interval: %.1fs  |  %d muestras",
             sample_rate, flow_interval, total_samples)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("No se pudo abrir el video: %s", video_path)
        sys.exit(1)

    # Leer todos los frames necesarios de manera eficiente
    # En lugar de leer frame a frame, usamos seek para saltar directo a los timestamps
    # Esto es mucho más rápido para videos largos en NAS

    raw_entries: list[dict] = []
    motion_count = 0

    log.info("Analizando movimiento...")

    for sample_idx in range(total_samples):
        t_a = sample_idx / sample_rate
        t_b = t_a + flow_interval

        if t_b >= dur:
            break

        # Seek a frame A
        cap.set(cv2.CAP_PROP_POS_MSEC, t_a * 1000)
        ret_a, frame_a = cap.read()
        if not ret_a:
            break

        # Seek a frame B
        cap.set(cv2.CAP_PROP_POS_MSEC, t_b * 1000)
        ret_b, frame_b = cap.read()
        if not ret_b:
            break

        # Reducir resolución para acelerar el cálculo del flow
        # El tracking no necesita 1920px — 480px es suficiente
        scale    = 480 / src_w
        small_a  = cv2.resize(frame_a, (480, int(src_h * scale)))
        small_b  = cv2.resize(frame_b, (480, int(src_h * scale)))

        center_x_small = detect_motion_center_x(
            small_a, small_b, min_motion_px, cv2, np
        )

        if center_x_small is not None:
            # Escalar de vuelta a coordenadas originales
            center_x = int(center_x_small / scale)
            crop_x   = center_to_crop_x(center_x, crop_w, src_w)
            detected = True
            motion_count += 1
        else:
            crop_x   = None   # sin movimiento — se rellenará
            detected = False

        raw_entries.append({
            "t":             round(t_a, 3),
            "crop_x":        crop_x,
            "face_detected": detected,   # True = movimiento detectado
        })

        # Progreso
        if total_samples > 0 and (sample_idx + 1) % max(1, total_samples // 10) == 0:
            pct = (sample_idx + 1) / total_samples * 100
            log.info("  %d%%  (%d/%d muestras | %d con movimiento)",
                     int(pct), sample_idx + 1, total_samples, motion_count)

    cap.release()

    n = len(raw_entries)
    detection_rate = motion_count / max(1, n) * 100

    log.info("Análisis completo: %d muestras | %.0f%% con movimiento detectado",
             n, detection_rate)

    if detection_rate < 20:
        log.warning(
            "Movimiento detectado en menos del 20%% de los frames. "
            "Probá bajar --min-motion (actual: %.1f) si el pastor se mueve poco.",
            min_motion_px
        )

    # ── Post-procesado ──

    # 1. Extraer valores raw (None donde no hubo movimiento)
    raw_x = [e["crop_x"] for e in raw_entries]

    # 2. Rellenar gaps
    log.info("Rellenando gaps e interpolando...")
    filled = fill_gaps(raw_x, fallback_x)

    # 3. EMA bidireccional
    log.info("Aplicando EMA bidireccional (alpha=%.2f)...", ema_alpha)
    smoothed_float = apply_ema_bidirectional(filled, ema_alpha)

    # 4. Límite de velocidad
    timestamps = [e["t"] for e in raw_entries]
    log.info("Aplicando límite de velocidad (%.0fpx/seg)...", max_pan_speed)
    final_xs = apply_speed_limit(
        smoothed_float, timestamps, max_pan_speed, src_w, crop_w
    )

    # 5. Construir entries finales
    entries = [
        {
            "t":             raw_entries[i]["t"],
            "crop_x":        final_xs[i],
            "face_detected": raw_entries[i]["face_detected"],
        }
        for i in range(n)
    ]

    return {
        "video":               video_path.name,
        "tracker":             "optical_flow",
        "src_width":           src_w,
        "src_height":          src_h,
        "crop_width":          crop_w,
        "sample_rate":         sample_rate,
        "flow_interval":       flow_interval,
        "min_motion_px":       min_motion_px,
        "ema_alpha":           ema_alpha,
        "max_pan_speed":       max_pan_speed,
        "duration":            round(dur, 3),
        "detection_rate_pct":  round(detection_rate, 1),
        "entries":             entries,
    }


# ──────────────────────────────────────────────
# QUERY HELPER (usado por generate_shorts.py)
# ──────────────────────────────────────────────

def get_crop_x_at(crop_map: dict, t_sec: float) -> int:
    entries = crop_map["entries"]
    if not entries:
        return (crop_map["src_width"] - crop_map["crop_width"]) // 2

    if t_sec <= entries[0]["t"]:
        return entries[0]["crop_x"]
    if t_sec >= entries[-1]["t"]:
        return entries[-1]["crop_x"]

    lo, hi = 0, len(entries) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if entries[mid]["t"] <= t_sec:
            lo = mid
        else:
            hi = mid

    t0, x0 = entries[lo]["t"], entries[lo]["crop_x"]
    t1, x1 = entries[hi]["t"], entries[hi]["crop_x"]
    if t1 == t0:
        return x0

    return int(x0 + (x1 - x0) * (t_sec - t0) / (t1 - t0))


# ──────────────────────────────────────────────
# ARG PARSER
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Genera crop_map.json con optical flow tracking (cámara fija)."
    )
    p.add_argument("--video",           required=True)
    p.add_argument("--output",          required=True)
    p.add_argument("--sample-rate",     type=int,   default=1,
                   help="Frames a analizar por segundo (default: 1)")
    p.add_argument("--flow-interval",   type=float, default=0.5,
                   help="Segundos entre frame A y B del flow (default: 0.5)")
    p.add_argument("--min-motion",      type=float, default=2.0,
                   help="Píxeles mínimos de movimiento para considerar un punto activo (default: 2.0)")
    p.add_argument("--ema-alpha",       type=float, default=0.15,
                   help="Factor EMA 0-1 (default: 0.15 — suave). Subir para más respuesta.")
    p.add_argument("--max-pan-speed",   type=float, default=80.0,
                   help="Máximos px/seg que puede moverse el crop (default: 80)")
    p.add_argument("--confidence",      type=float, default=0.0,
                   help="No usado — solo por compatibilidad con v1")
    p.add_argument("--smooth-window",   type=int,   default=0,
                   help="No usado — solo por compatibilidad con v1")
    p.add_argument("--debug",           action="store_true")
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
    log.info("Face tracker v2  —  Optical Flow")
    log.info("Video:          %s", video_path)
    log.info("Sample rate:    %d fps", args.sample_rate)
    log.info("Flow interval:  %.1fs", args.flow_interval)
    log.info("Min motion:     %.1f px", args.min_motion)
    log.info("EMA alpha:      %.2f", args.ema_alpha)
    log.info("Max pan speed:  %.0f px/s", args.max_pan_speed)
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    crop_map = analyze_video(
        video_path    = video_path,
        sample_rate   = args.sample_rate,
        flow_interval = args.flow_interval,
        min_motion_px = args.min_motion,
        ema_alpha     = args.ema_alpha,
        max_pan_speed = args.max_pan_speed,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(crop_map, f, indent=2, ensure_ascii=False)

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("Guardado:       %s", output_path)
    log.info("Entradas:       %d", len(crop_map["entries"]))
    log.info("Movimiento:     %.1f%%", crop_map["detection_rate_pct"])
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
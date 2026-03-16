#!/usr/bin/env python3
"""
face_tracker.py
===============
Analiza un video y genera un crop_map.json con la posición X del rostro
por cada segundo, suavizada con interpolación para crop dinámico fluido.

Compatible con MediaPipe >= 0.10 (Tasks API).

Uso:
    python face_tracker.py \
        --video  /mnt/nas/predicas/20260315_wxM8MkMKvNE.mp4 \
        --output ./output/shorts/crop_map.json \
        [--sample-rate 1]     # frames a analizar por segundo (default: 1)
        [--smooth-window 3]   # segundos de ventana de suavizado (default: 3)
        [--confidence 0.4]    # confianza mínima detección (default: 0.4)
        [--debug]

Dependencias:
    pip install mediapipe opencv-python-headless

El modelo blaze_face_short_range.tflite se descarga automáticamente
la primera vez (~1MB desde storage.googleapis.com).
"""

import argparse
import json
import logging
import math
import subprocess
import sys
import urllib.request
from pathlib import Path

log = logging.getLogger("face_tracker")

# ──────────────────────────────────────────────
# MODELO
# ──────────────────────────────────────────────

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)
MODEL_NAME = "blaze_face_short_range.tflite"


def get_model_path() -> Path:
    model_path = Path(__file__).parent / MODEL_NAME
    if model_path.exists():
        return model_path
    log.info("Descargando modelo MediaPipe (~1MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        log.info("Modelo descargado: %s", model_path)
    except Exception as e:
        log.error(
            "No se pudo descargar el modelo: %s\n"
            "Descargalo manualmente desde:\n  %s\n"
            "y copialo junto a face_tracker.py.",
            e, MODEL_URL
        )
        sys.exit(1)
    return model_path


# ──────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────

def import_deps():
    """Importa cv2 y mediapipe Tasks API con mensajes claros."""
    try:
        import cv2
    except ImportError:
        log.error("opencv no instalado.\nInstalalo: pip install opencv-python-headless")
        sys.exit(1)

    try:
        import mediapipe as mp
    except ImportError:
        log.error("mediapipe no instalado.\nInstalalo: pip install mediapipe")
        sys.exit(1)

    version = tuple(int(x) for x in mp.__version__.split(".")[:2])
    log.info("MediaPipe %s", mp.__version__)

    if version < (0, 10):
        log.error(
            "Se requiere MediaPipe >= 0.10 (instalado: %s).\n"
            "Actualizá: pip install -U mediapipe",
            mp.__version__
        )
        sys.exit(1)

    # Tasks API — disponible desde 0.10
    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
    except ImportError as e:
        log.error(
            "Error importando MediaPipe Tasks API: %s\n"
            "Reinstalá: pip install -U mediapipe",
            e
        )
        sys.exit(1)

    return cv2, mp, BaseOptions, FaceDetector, FaceDetectorOptions, RunningMode


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
# HELPERS DE CROP
# ──────────────────────────────────────────────

def compute_crop_width(src_w: int, src_h: int) -> int:
    return min(math.floor(src_h * 9 / 16), src_w)


def center_to_crop_x(center_x: int, crop_w: int, src_w: int) -> int:
    return max(0, min(center_x - crop_w // 2, src_w - crop_w))


# ──────────────────────────────────────────────
# SUAVIZADO
# ──────────────────────────────────────────────

def smooth_crop_positions(entries: list[dict], window_sec: int) -> list[dict]:
    n = len(entries)
    if n == 0:
        return entries

    detected = [i for i, e in enumerate(entries) if e["face_detected"]]
    if not detected:
        log.warning("Ningún rostro detectado — usando crop centrado fijo.")
        return entries

    # Paso 1: rellenar gaps por interpolación lineal entre detecciones
    filled = [e["crop_x"] for e in entries]
    for i in range(n):
        if not entries[i]["face_detected"]:
            prev = next((j for j in range(i - 1, -1, -1) if entries[j]["face_detected"]), None)
            nxt  = next((j for j in range(i + 1, n)      if entries[j]["face_detected"]), None)
            if prev is not None and nxt is not None:
                t          = (i - prev) / (nxt - prev)
                filled[i]  = int(filled[prev] + t * (filled[nxt] - filled[prev]))
            elif prev is not None:
                filled[i] = filled[prev]
            elif nxt is not None:
                filled[i] = filled[nxt]

    # Paso 2: ventana deslizante centrada
    half     = max(1, window_sec // 2)
    smoothed = [
        int(sum(filled[max(0, i - half): min(n, i + half + 1)]) /
            len(filled[max(0, i - half): min(n, i + half + 1)]))
        for i in range(n)
    ]

    return [
        {
            "t":             entries[i]["t"],
            "crop_x":        smoothed[i],
            "face_detected": entries[i]["face_detected"],
        }
        for i in range(n)
    ]


# ──────────────────────────────────────────────
# ANÁLISIS PRINCIPAL
# ──────────────────────────────────────────────

def analyze_video(
    video_path:    Path,
    sample_rate:   int,
    smooth_window: int,
    confidence:    float,
) -> dict:

    cv2, mp, BaseOptions, FaceDetector, FaceDetectorOptions, RunningMode = import_deps()

    info    = get_video_info(video_path)
    src_w   = info["width"]
    src_h   = info["height"]
    fps     = info["fps"]
    dur     = info["duration"]
    crop_w  = compute_crop_width(src_w, src_h)

    frame_interval = max(1, int(fps / sample_rate))
    total_samples  = int(dur * sample_rate)

    log.info("Video: %dx%d  %.1fs  %.1f fps", src_w, src_h, dur, fps)
    log.info("Crop 9:16: %dpx ancho", crop_w)
    log.info("Muestras: %d (cada %d frames)", total_samples, frame_interval)

    model_path = get_model_path()

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        min_detection_confidence=confidence,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("No se pudo abrir el video: %s", video_path)
        sys.exit(1)

    entries        = []
    frame_idx      = 0
    sample_idx     = 0
    detected_count = 0

    with FaceDetector.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            t_sec = frame_idx / fps

            # BGR → RGB → mediapipe Image
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result   = detector.detect(mp_image)
            detected = bool(result.detections)

            if detected:
                # Detección con mayor score
                best     = max(result.detections, key=lambda d: d.categories[0].score)
                bbox     = best.bounding_box   # origin_x, origin_y, width, height en px
                center_x = bbox.origin_x + bbox.width // 2
                crop_x   = center_to_crop_x(center_x, crop_w, src_w)
                detected_count += 1
            else:
                crop_x = (src_w - crop_w) // 2  # placeholder — se interpolará

            entries.append({
                "t":             round(t_sec, 3),
                "crop_x":        crop_x,
                "face_detected": detected,
            })

            sample_idx += 1
            if total_samples > 0 and sample_idx % max(1, total_samples // 10) == 0:
                log.info(
                    "  %d%%  (%d/%d muestras | %d rostros detectados)",
                    int(sample_idx / total_samples * 100),
                    sample_idx, total_samples, detected_count,
                )

            frame_idx += 1

    cap.release()

    detection_rate = detected_count / max(1, len(entries)) * 100
    log.info("Análisis completo: %d muestras | %.0f%% detección", len(entries), detection_rate)

    if detection_rate < 30:
        log.warning(
            "Tasa de detección baja (%.0f%%). "
            "Probá --confidence 0.3 o revisá la iluminación del video.",
            detection_rate
        )

    log.info("Suavizando (ventana %ds)...", smooth_window)
    smoothed = smooth_crop_positions(entries, smooth_window)

    return {
        "video":               video_path.name,
        "src_width":           src_w,
        "src_height":          src_h,
        "crop_width":          crop_w,
        "sample_rate":         sample_rate,
        "smooth_window":       smooth_window,
        "duration":            round(dur, 3),
        "detection_rate_pct":  round(detection_rate, 1),
        "entries":             smoothed,
    }


# ──────────────────────────────────────────────
# QUERY HELPER (importado por generate_shorts.py)
# ──────────────────────────────────────────────

def get_crop_x_at(crop_map: dict, t_sec: float) -> int:
    """Interpolación lineal del crop_x para un tiempo dado."""
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
        description="Genera crop_map.json con face tracking para clips 9:16."
    )
    p.add_argument("--video",          required=True)
    p.add_argument("--output",         required=True)
    p.add_argument("--sample-rate",    type=int,   default=1)
    p.add_argument("--smooth-window",  type=int,   default=3)
    p.add_argument("--confidence",     type=float, default=0.4,
                   help="Confianza mínima de detección 0.0-1.0 (default: 0.4)")
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
    log.info("Face tracker  (MediaPipe >= 0.10)")
    log.info("Video:         %s", video_path)
    log.info("Sample rate:   %d fps", args.sample_rate)
    log.info("Smooth window: %ds", args.smooth_window)
    log.info("Confidence:    %.1f", args.confidence)
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    crop_map = analyze_video(
        video_path    = video_path,
        sample_rate   = args.sample_rate,
        smooth_window = args.smooth_window,
        confidence    = args.confidence,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(crop_map, f, indent=2, ensure_ascii=False)

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("Guardado: %s", output_path)
    log.info("Entradas: %d | Detección: %.1f%%",
             len(crop_map["entries"]), crop_map["detection_rate_pct"])
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()

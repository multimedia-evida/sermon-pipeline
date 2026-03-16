#!/usr/bin/env python3
"""
preview_crop.py
===============
Genera una imagen PNG mostrando exactamente dónde queda el crop 9:16
para un segundo dado del video, con y sin offset.

Útil para calibrar --crop-offset sin tener que procesar clips completos.

Uso:
    python preview_crop.py \
        --video    /mnt/nas/predicas/20260315_wxM8MkMKvNE.mp4 \
        --crop-map ./data/output/shorts/20260315_wxM8MkMKvNE_crop_map.json \
        --time     90 \
        --offset   250 \
        --output   ./preview.png

    # Sin crop_map (solo muestra el crop centrado):
    python preview_crop.py --video video.mp4 --time 90 --output preview.png

    # Probar varios offsets de una vez:
    python preview_crop.py --video video.mp4 --time 90 --offset 0 100 200 300 --output preview.png

Dependencias:
    pip install opencv-python-headless
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        print("ERROR: opencv no instalado. Instalalo con: pip install opencv-python-headless")
        sys.exit(1)


def extract_frame(video_path: Path, time_sec: float) -> "cv2.Mat":
    """Extrae un frame del video en el tiempo dado usando ffmpeg → pipe."""
    cmd = [
        "ffmpeg", "-ss", str(time_sec),
        "-i", str(video_path),
        "-frames:v", "1",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(f"ERROR ffmpeg: {r.stderr.decode()[-500:]}")
        sys.exit(1)

    cv2 = import_cv2()
    import numpy as np

    # Obtener dimensiones del video
    probe = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path),
    ], capture_output=True, text=True)
    w, h = probe.stdout.strip().split(",")
    w, h = int(w), int(h)

    frame = np.frombuffer(r.stdout, dtype=np.uint8).reshape((h, w, 3))
    return frame, w, h


def compute_crop_width(src_w: int, src_h: int) -> int:
    return min(math.floor(src_h * 9 / 16), src_w)


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


def draw_crop_overlay(frame, crop_x: int, crop_w: int, src_h: int, offset: int, color, label: str):
    """Dibuja el rectángulo de crop y su label sobre el frame."""
    cv2 = import_cv2()

    # Oscurecer lo que queda fuera del crop
    overlay = frame.copy()

    # Rectángulo del crop
    x1 = max(0, crop_x)
    x2 = min(frame.shape[1], crop_x + crop_w)
    cv2.rectangle(overlay, (x1, 0), (x2, src_h), color, 3)

    # Label
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness  = 2
    text       = f"offset={offset:+d}px  x={crop_x}"
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = max(5, x1 + 10)
    ty = 60
    cv2.rectangle(overlay, (tx - 5, ty - th - 10), (tx + tw + 5, ty + 10), (0, 0, 0), -1)
    cv2.putText(overlay, text, (tx, ty), font, font_scale, color, thickness)

    # Línea central del crop (para ver dónde apunta)
    cx = crop_x + crop_w // 2
    cv2.line(overlay, (cx, 0), (cx, src_h), color, 1)

    return overlay


def main():
    p = argparse.ArgumentParser(
        description="Preview visual del crop 9:16 para calibrar --crop-offset."
    )
    p.add_argument("--video",    required=True, help="Video de entrada")
    p.add_argument("--crop-map", default=None,  help="crop_map.json del face tracker")
    p.add_argument("--time",     type=float, default=60.0,
                   help="Segundo del video a previsualizar (default: 60)")
    p.add_argument("--offset",   type=int, nargs="+", default=[0],
                   help="Uno o más valores de offset a comparar (ej: 0 100 200 300)")
    p.add_argument("--output",   default="./preview.png",
                   help="Ruta del PNG de salida (default: ./preview.png)")
    args = p.parse_args()

    cv2 = import_cv2()
    import numpy as np

    video_path = Path(args.video)
    output     = Path(args.output)

    if not video_path.exists():
        print(f"ERROR: video no encontrado: {video_path}")
        sys.exit(1)

    # Extraer frame
    print(f"Extrayendo frame en t={args.time}s...")
    frame, src_w, src_h = extract_frame(video_path, args.time)
    crop_w = compute_crop_width(src_w, src_h)

    # Cargar crop_map si existe
    crop_map = None
    if args.crop_map and Path(args.crop_map).exists():
        with open(args.crop_map, encoding="utf-8") as f:
            crop_map = json.load(f)
        base_crop_x = get_crop_x_at(crop_map, args.time)
        print(f"crop_map cargado — posición base del tracker en t={args.time}s: x={base_crop_x}")
    else:
        base_crop_x = (src_w - crop_w) // 2
        print(f"Sin crop_map — usando crop centrado: x={base_crop_x}")

    # Colores para cada offset (BGR)
    colors = [
        (0, 255, 0),    # verde
        (0, 165, 255),  # naranja
        (255, 0, 0),    # azul
        (0, 0, 255),    # rojo
        (255, 0, 255),  # magenta
    ]

    offsets = args.offset
    n_cols  = len(offsets)

    # Escalar frame para que quede legible en el preview
    scale      = min(1.0, 1200 / src_w)
    disp_w     = int(src_w * scale)
    disp_h     = int(src_h * scale)
    frame_disp = cv2.resize(frame, (disp_w, disp_h))

    # Generar una columna por offset
    cols = []
    for i, offset in enumerate(offsets):
        crop_x     = base_crop_x + offset
        crop_x     = max(0, min(crop_x, src_w - crop_w))
        color      = colors[i % len(colors)]
        annotated  = draw_crop_overlay(
            frame.copy(), crop_x, crop_w, src_h, offset, color,
            label=f"offset={offset:+d}"
        )
        col = cv2.resize(annotated, (disp_w, disp_h))

        # Oscurecer zonas fuera del crop para mejor visualización
        mask          = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        x1_s          = int(crop_x * scale)
        x2_s          = int((crop_x + crop_w) * scale)
        mask[:, x1_s:x2_s] = col[:, x1_s:x2_s]
        outside       = col.copy()
        outside       = (outside * 0.4).astype(np.uint8)
        outside[:, x1_s:x2_s] = col[:, x1_s:x2_s]
        col           = outside

        # Redibujar el rectángulo encima del oscurecido
        cv2.rectangle(col, (x1_s, 0), (x2_s, disp_h - 1), color, 3)
        cx_s = int((crop_x + crop_w // 2) * scale)
        cv2.line(col, (cx_s, 0), (cx_s, disp_h), color, 1)

        # Label grande
        label_text = f"offset={offset:+d}  crop_x={crop_x}"
        cv2.putText(col, label_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cols.append(col)

    # Agregar separadores verticales
    sep    = np.zeros((disp_h, 4, 3), dtype=np.uint8)
    result = cols[0]
    for col in cols[1:]:
        result = np.hstack([result, sep, col])

    # Info al pie
    info_h  = 60
    info    = np.zeros((info_h, result.shape[1], 3), dtype=np.uint8)
    info_txt = (
        f"Video: {video_path.name}  |  t={args.time}s  |  "
        f"src={src_w}x{src_h}  |  crop_w={crop_w}px  |  "
        f"tracker_base_x={base_crop_x}"
    )
    cv2.putText(info, info_txt, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    final = np.vstack([result, info])

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), final)

    print(f"\n✓ Preview guardado en: {output}")
    print(f"  Resolución: {final.shape[1]}x{final.shape[0]}px")
    print(f"\nOffsets probados:")
    for i, offset in enumerate(offsets):
        cx = base_crop_x + offset
        cx = max(0, min(cx, src_w - crop_w))
        print(f"  offset={offset:+5d}px  →  crop_x={cx}  (centro en x={cx + crop_w // 2})")


if __name__ == "__main__":
    main()

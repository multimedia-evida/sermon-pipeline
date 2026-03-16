#!/usr/bin/env python3
"""
generate_shorts.py
==================
Genera clips verticales 9:16 (Shorts/Reels) desde un video horizontal,
usando los key_moments del JSON procesado por IA y los segmentos de
Whisper como subtítulos.

Uso:
    python generate_shorts.py \
        --predica   dBp8VW3rAdQ_Pure_De_Papas.json \
        --whisper   20250721_dBp8VW3rAdQ_Pure_De_Papas.json \
        --video     video.mp4 \
        --output    ./output \
        [--resolution 1080x1920] \
        [--crop-offset 0] \
        [--no-concat]

Dependencias:
    - ffmpeg >= 4.4 (en PATH)
    - Python 3.8+
    - No requiere librerías externas de Python
"""

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_shorts")


@dataclass
class Config:
    predica_json: Path
    whisper_json: Path
    video_path: Path
    output_dir: Path
    width: int = 1080
    height: int = 1920
    crop_offset_x: int = 0          # offset horizontal manual (px sobre el video original)
    concat: bool = True             # generar también un video compilado
    font_size: int = 60             # tamaño base de fuente para subtítulos
    font_color: str = "white"
    font_outline_color: str = "black"
    font_outline_width: int = 4
    subtitle_margin_bottom: int = 200   # px desde el fondo del frame vertical
    max_chars_per_line: int = 35        # wrap de texto


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def check_ffmpeg() -> None:
    """Verifica que ffmpeg esté disponible en PATH."""
    if not shutil.which("ffmpeg"):
        log.error("ffmpeg no encontrado en PATH. Instalalo con: apt install ffmpeg")
        sys.exit(1)
    result = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True
    )
    version_line = result.stdout.splitlines()[0]
    log.info("ffmpeg encontrado: %s", version_line)


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """Devuelve (width, height) del video usando ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ffprobe falló: %s", result.stderr)
        sys.exit(1)
    w, h = result.stdout.strip().split("x")
    return int(w), int(h)


def load_predica_json(path: Path) -> dict:
    """Carga el JSON procesado por IA. Soporta lista o dict."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # El JSON puede ser una lista con un único elemento (formato n8n)
    if isinstance(data, list):
        data = data[0]
    # Navegar hasta output si existe
    if "output" in data:
        data = data["output"]
    return data


def load_whisper_json(path: Path) -> list[dict]:
    """
    Carga los segmentos de Whisper.
    Soporta:
      - {"segments": [...]}        ← formato del proyecto
      - [{"start":..., "text":...}]  ← lista directa
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    if isinstance(data, list):
        return data
    log.error("Formato de Whisper JSON no reconocido.")
    sys.exit(1)


def escape_drawtext(text: str) -> str:
    """
    Escapa caracteres especiales para el filtro drawtext de ffmpeg.
    Referencia: https://ffmpeg.org/ffmpeg-filters.html#drawtext
    """
    # Orden importa: primero backslash
    replacements = [
        ("\\", "\\\\"),
        ("'",  "\u2019"),   # reemplaza comilla simple por curva (más seguro)
        (":",  "\\:"),
        ("%",  "\\%"),
        ("\n", " "),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def wrap_text(text: str, max_chars: int) -> str:
    """
    Hace word-wrap básico insertando \\n para ffmpeg drawtext.
    Retorna el texto con saltos de línea escapados para drawtext.
    """
    words = text.split()
    lines = []
    current = []
    current_len = 0

    for word in words:
        if current_len + len(word) + len(current) > max_chars:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word)

    if current:
        lines.append(" ".join(current))

    return "\\n".join(lines)


def segments_for_range(
    segments: list[dict],
    start_sec: float,
    end_sec: float,
) -> list[dict]:
    """
    Filtra y re-temporiza segmentos de Whisper para un rango dado.
    Los timestamps resultantes son relativos al inicio del clip (t=0).
    """
    result = []
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Descarte si no hay overlap
        if seg_end <= start_sec or seg_start >= end_sec:
            continue

        # Clamp al rango del clip
        clipped_start = max(seg_start, start_sec) - start_sec
        clipped_end = min(seg_end, end_sec) - start_sec

        if clipped_end - clipped_start < 0.1:  # segmento demasiado corto
            continue

        result.append({
            "start": round(clipped_start, 3),
            "end": round(clipped_end, 3),
            "text": seg["text"].strip(),
        })

    return result


def build_drawtext_filters(
    segments: list[dict],
    cfg: Config,
    frame_width: int,
    frame_height: int,
) -> list[str]:
    """
    Genera una lista de filtros drawtext, uno por segmento.
    Cada filtro muestra el texto solo durante su ventana temporal.
    """
    filters = []
    y_pos = frame_height - cfg.subtitle_margin_bottom

    for seg in segments:
        text = wrap_text(escape_drawtext(seg["text"]), cfg.max_chars_per_line)
        t_start = seg["start"]
        t_end = seg["end"]

        # Sombra/outline: se logra con dos drawtext superpuestos (outline + fill)
        # Primero el outline (negro), luego el texto encima (blanco)
        base = (
            f"drawtext="
            f"text='{text}':"
            f"fontsize={cfg.font_size}:"
            f"fontcolor={cfg.font_color}:"
            f"borderw={cfg.font_outline_width}:"
            f"bordercolor={cfg.font_outline_color}:"
            f"x=(w-text_w)/2:"
            f"y={y_pos}-text_h:"
            f"line_spacing=8:"
            f"enable='between(t,{t_start},{t_end})'"
        )
        filters.append(base)

    return filters


def build_ffmpeg_cmd(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    drawtext_filters: list[str],
    cfg: Config,
    src_width: int,
    src_height: int,
) -> list[str]:
    """
    Construye el comando ffmpeg completo para un clip.

    Pipeline de filtros:
      1. trim + setpts        → recorte temporal
      2. crop                 → recorte vertical centrado (o con offset)
      3. scale                → escala al tamaño de salida
      4. drawtext (×N)        → subtítulos por segmento
    """
    duration = end_sec - start_sec

    # ── Crop: recorte vertical centrado en el eje X ──
    # Queremos un crop con aspect ratio 9:16 del video original.
    # Tomamos toda la altura y ajustamos el ancho.
    crop_h = src_height
    crop_w = math.floor(src_height * 9 / 16)

    # Si el video original es más angosto que el crop necesario, invertimos:
    # tomamos todo el ancho y limitamos la altura.
    if crop_w > src_width:
        crop_w = src_width
        crop_h = math.floor(src_width * 16 / 9)

    # Centro horizontal con offset configurable
    crop_x = max(0, (src_width - crop_w) // 2 + cfg.crop_offset_x)
    crop_x = min(crop_x, src_width - crop_w)  # no salirse del borde
    crop_y = (src_height - crop_h) // 2

    # ── Filtro completo ──
    filter_parts = [
        f"trim=start={start_sec}:end={end_sec}",
        "setpts=PTS-STARTPTS",
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
        f"scale={cfg.width}:{cfg.height}:flags=lanczos",
    ] + drawtext_filters

    filter_complex = ",".join(filter_parts)

    # Audio: trim independiente
    audio_filter = (
        f"[0:a]atrim=start={start_sec}:end={end_sec},"
        f"asetpts=PTS-STARTPTS[aout]"
    )

    cmd = [
        "ffmpeg",
        "-y",                           # sobreescribir sin preguntar
        "-i", str(video_path),
        "-filter_complex",
        f"[0:v]{filter_complex}[vout];{audio_filter}",
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",                   # calidad visual alta
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",      # streaming-friendly
        "-t", str(duration),
        str(output_path),
    ]

    return cmd


def run_ffmpeg(cmd: list[str], label: str) -> bool:
    """Ejecuta ffmpeg, loguea stderr en caso de error. Retorna True si OK."""
    log.info("Procesando: %s", label)
    log.debug("CMD: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error("ffmpeg falló para '%s':\n%s", label, result.stderr[-2000:])
        return False

    log.info("✓ Generado: %s", label)
    return True


def concat_clips(clip_paths: list[Path], output_path: Path) -> bool:
    """
    Concatena una lista de clips usando el demuxer concat de ffmpeg.
    Más rápido que re-encodear si todos tienen el mismo codec/resolución.
    """
    if not clip_paths:
        return False

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        concat_file = f.name
        for p in clip_paths:
            f.write(f"file '{p.resolve()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c", "copy",               # copy stream, sin re-encode
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(concat_file)

    if result.returncode != 0:
        log.error("Concat falló:\n%s", result.stderr[-2000:])
        return False

    log.info("✓ Compilado generado: %s", output_path)
    return True


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Genera clips 9:16 con subtítulos desde key_moments de prédica."
    )
    parser.add_argument("--predica",   required=True, help="JSON procesado por IA")
    parser.add_argument("--whisper",   required=True, help="JSON de transcripción Whisper")
    parser.add_argument("--video",     required=True, help="Video original (.mp4)")
    parser.add_argument("--output",    default="./output", help="Directorio de salida")
    parser.add_argument(
        "--resolution", default="1080x1920",
        help="Resolución de salida WxH (default: 1080x1920)"
    )
    parser.add_argument(
        "--crop-offset", type=int, default=0,
        help="Offset X del crop en px sobre el video original (+ derecha, - izquierda)"
    )
    parser.add_argument(
        "--no-concat", action="store_true",
        help="No generar video compilado con todos los clips"
    )
    parser.add_argument(
        "--font-size", type=int, default=60,
        help="Tamaño de fuente para subtítulos (default: 60)"
    )
    parser.add_argument(
        "--subtitle-margin", type=int, default=200,
        help="Margen inferior de subtítulos en px (default: 200)"
    )
    parser.add_argument("--debug", action="store_true", help="Logging verboso")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    w, h = args.resolution.split("x")

    return Config(
        predica_json=Path(args.predica),
        whisper_json=Path(args.whisper),
        video_path=Path(args.video),
        output_dir=Path(args.output),
        width=int(w),
        height=int(h),
        crop_offset_x=args.crop_offset,
        concat=not args.no_concat,
        font_size=args.font_size,
        subtitle_margin_bottom=args.subtitle_margin,
    )


def main() -> None:
    cfg = parse_args()

    # ── Validaciones ──
    check_ffmpeg()

    for p, label in [
        (cfg.predica_json, "--predica"),
        (cfg.whisper_json, "--whisper"),
        (cfg.video_path,   "--video"),
    ]:
        if not p.exists():
            log.error("Archivo no encontrado (%s): %s", label, p)
            sys.exit(1)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar datos ──
    predica = load_predica_json(cfg.predica_json)
    segments = load_whisper_json(cfg.whisper_json)
    key_moments = predica.get("key_moments", [])

    if not key_moments:
        log.error("No se encontraron key_moments en el JSON de prédica.")
        sys.exit(1)

    log.info("Video: %s", cfg.video_path)
    log.info("Key moments encontrados: %d", len(key_moments))

    # ── Dimensiones del video fuente ──
    src_w, src_h = get_video_dimensions(cfg.video_path)
    log.info("Resolución fuente: %dx%d", src_w, src_h)

    # ── Procesar cada key_moment ──
    generated_clips: list[Path] = []

    for km in key_moments:
        km_id    = km["id"]
        title    = km["title"]
        start    = float(km["start_seconds"])
        end      = float(km["end_seconds"])

        # Nombre de archivo seguro
        safe_title = "".join(
            c if c.isalnum() or c in (" ", "-", "_") else "_"
            for c in title
        ).strip().replace(" ", "_")[:50]

        clip_filename = f"{km_id:02d}_{safe_title}.mp4"
        clip_path = cfg.output_dir / clip_filename

        log.info(
            "── Key moment %d/%d: '%s' [%.1fs → %.1fs]",
            km_id, len(key_moments), title, start, end
        )

        # Segmentos de Whisper para este rango
        clip_segments = segments_for_range(segments, start, end)
        log.info("   Segmentos de subtítulos: %d", len(clip_segments))

        if not clip_segments:
            log.warning("   Sin segmentos Whisper para este rango, se genera sin subtítulos.")

        # Filtros drawtext
        drawtext = build_drawtext_filters(
            clip_segments, cfg, cfg.width, cfg.height
        )

        # Comando ffmpeg
        cmd = build_ffmpeg_cmd(
            video_path=cfg.video_path,
            output_path=clip_path,
            start_sec=start,
            end_sec=end,
            drawtext_filters=drawtext,
            cfg=cfg,
            src_width=src_w,
            src_height=src_h,
        )

        ok = run_ffmpeg(cmd, f"clip_{km_id:02d}_{safe_title}")

        if ok:
            generated_clips.append(clip_path)
        else:
            log.warning("   Clip %d omitido del compilado por error.", km_id)

    # ── Compilado ──
    if cfg.concat and len(generated_clips) > 1:
        concat_path = cfg.output_dir / "00_compilado_completo.mp4"
        log.info("Generando compilado con %d clips...", len(generated_clips))
        concat_clips(generated_clips, concat_path)

    # ── Resumen ──
    log.info("══════════════════════════════")
    log.info("Clips generados: %d / %d", len(generated_clips), len(key_moments))
    log.info("Directorio de salida: %s", cfg.output_dir.resolve())
    log.info("══════════════════════════════")


if __name__ == "__main__":
    main()

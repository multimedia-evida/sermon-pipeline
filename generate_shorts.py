#!/usr/bin/env python3
"""
generate_shorts.py
==================
Genera clips verticales 9:16 (Shorts/Reels) desde un video horizontal,
usando los key_moments del JSON procesado por IA y los segmentos de
Whisper como subtítulos.

Uso:
    python generate_shorts.py \
        --predica   ~/repo/sermon-pipeline/data/input/ai-json/'wxM8MkMKvNE_¿Podrías vivir sin Dios_ - Pr. Beto Tassara.json' \
        --whisper   /home/evida/repo/sermon-pipeline/data/output/transcripts/done/20260315_wxM8MkMKvNE.json \
        --video     /mnt/nas/predicas/20260315_wxM8MkMKvNE.mp4 \
        --output    ./output/shorts \
        [--resolution 1080x1920] \
        [--crop-offset 0] \
        [--no-concat]

Opcionales:
    --resolution    1080x1920
    --no-concat
    --font-size     72
    --subtitle-y    0.72
    --max-chars     28
    --crop-map      ./output/shorts/crop_map.json   (ruta custom al caché)
    --no-tracking                                   (deshabilita face tracking)
    --tracker-sample-rate   1                       (fps de análisis del tracker)
    --tracker-smooth-window 3                       (segundos de suavizado)
    --debug

Dependencias:
    - ffmpeg >= 4.4 con libass
    - Python 3.8+
    - mediapipe + opencv-python-headless  (solo para face tracking)

Instalar tracking:
    pip install mediapipe opencv-python-headless
"""

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_shorts")


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

@dataclass
class Config:
    predica_json:           Path
    whisper_path:           Path
    video_path:             Path
    output_dir:             Path
    width:                  int   = 1080
    height:                 int   = 1920
    concat:                 bool  = True
    font_size:              int   = 72
    subtitle_y_pct:         float = 0.72
    max_chars_per_line:     int   = 28
    # Face tracking
    use_tracking:           bool  = True
    crop_map_path:          Path  = None     # None → auto-resolve
    tracker_sample_rate:    int   = 1
    tracker_smooth_window:  int   = 3


# ──────────────────────────────────────────────
# SAFE ZONE REFERENCE
# ──────────────────────────────────────────────
#
#  1920px
#  ┌──────────────────────────┐ 0
#  │  TOP UNSAFE   (0-8%)     │
#  ├──────────────────────────┤ ~154px
#  │   SAFE ZONE              │
#  ├──────────────────────────┤ ~1382px  ← subtitle_y_pct=0.72
#  │  SUBTÍTULOS              │
#  ├──────────────────────────┤ ~1536px
#  │  BOTTOM UNSAFE (80-100%) │ ← controles app
#  └──────────────────────────┘ 1920px


# ──────────────────────────────────────────────
# FFMPEG
# ──────────────────────────────────────────────

def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        log.error("ffmpeg no encontrado. Instalalo con: sudo apt install ffmpeg")
        sys.exit(1)
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    log.info("ffmpeg: %s", r.stdout.splitlines()[0])


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("ffprobe falló: %s", r.stderr)
        sys.exit(1)
    w, h = r.stdout.strip().split("x")
    return int(w), int(h)


def get_video_fps(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        str(video_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    num, den = r.stdout.strip().split("/")
    return float(num) / float(den)


# ──────────────────────────────────────────────
# JSON LOADERS
# ──────────────────────────────────────────────

def load_predica_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = data[0]
    if "output" in data:
        data = data["output"]
    return data


def extract_video_id(predica_path: Path) -> str:
    stem = predica_path.stem
    stem_clean = re.sub(r'^\d{8}_', '', stem)
    match = re.match(r'([A-Za-z0-9_\-]{11})', stem_clean)
    if match:
        return match.group(1)
    parts = stem.split("_")
    for p in parts:
        if len(p) == 11 and re.match(r'^[A-Za-z0-9_\-]+$', p):
            return p
    return parts[0]


def resolve_whisper_path(whisper_path: Path, predica_path: Path) -> Path:
    if whisper_path.is_file():
        return whisper_path
    if not whisper_path.is_dir():
        log.error("--whisper no es un archivo ni directorio: %s", whisper_path)
        sys.exit(1)
    video_id   = extract_video_id(predica_path)
    candidates = [p for p in whisper_path.glob("*.json") if video_id in p.stem]
    if not candidates:
        available = [p.name for p in whisper_path.glob("*.json")]
        log.error(
            "No se encontró Whisper JSON para videoId '%s' en %s\n"
            "Disponibles: %s",
            video_id, whisper_path,
            available if available else "(ninguno)"
        )
        sys.exit(1)
    if len(candidates) > 1:
        log.warning("Múltiples candidatos Whisper, usando: %s", candidates[0].name)
    log.info("Whisper JSON: %s", candidates[0].name)
    return candidates[0]


def load_whisper_segments(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    if isinstance(data, list):
        return data
    log.error("Formato Whisper JSON no reconocido: %s", path)
    sys.exit(1)


def segments_for_range(
    segments: list[dict], start_sec: float, end_sec: float
) -> list[dict]:
    result = []
    for seg in segments:
        s, e = float(seg["start"]), float(seg["end"])
        if e <= start_sec or s >= end_sec:
            continue
        cs = max(s, start_sec) - start_sec
        ce = min(e, end_sec)   - start_sec
        if ce - cs < 0.1:
            continue
        result.append({
            "start": round(cs, 3),
            "end":   round(ce, 3),
            "text":  seg["text"].strip(),
        })
    return result


# ──────────────────────────────────────────────
# FACE TRACKING — CACHÉ Y AUTO-INVOCACIÓN
# ──────────────────────────────────────────────

def resolve_crop_map_path(cfg: Config) -> Path:
    """Determina la ruta del crop_map.json (custom o auto-generada)."""
    if cfg.crop_map_path:
        return cfg.crop_map_path
    # Convención: mismo directorio de salida, nombre basado en el video
    stem = cfg.video_path.stem
    return cfg.output_dir / f"{stem}_crop_map.json"


def load_or_generate_crop_map(cfg: Config) -> dict | None:
    """
    Carga el crop_map.json si existe.
    Si no existe, invoca face_tracker.py para generarlo.
    Retorna None si el tracking está deshabilitado.
    """
    if not cfg.use_tracking:
        log.info("Face tracking deshabilitado (--no-tracking).")
        return None

    crop_map_path = resolve_crop_map_path(cfg)

    # ── Caché hit ──
    if crop_map_path.exists():
        log.info("Caché de tracking encontrado: %s", crop_map_path.name)
        with open(crop_map_path, encoding="utf-8") as f:
            crop_map = json.load(f)
        log.info(
            "  %d entradas · %.1f%% detección · suavizado %ds",
            len(crop_map.get("entries", [])),
            crop_map.get("detection_rate_pct", 0),
            crop_map.get("smooth_window", 0),
        )
        return crop_map

    # ── Caché miss → invocar face_tracker.py ──
    log.info("crop_map.json no encontrado. Invocando face_tracker.py...")

    tracker_script = Path(__file__).parent / "face_tracker.py"
    if not tracker_script.exists():
        log.error(
            "face_tracker.py no encontrado en %s\n"
            "Copialo junto a generate_shorts.py o usá --no-tracking.",
            tracker_script.parent
        )
        log.warning("Continuando sin face tracking (crop centrado).")
        return None

    cmd = [
        sys.executable,           # mismo intérprete Python del venv
        str(tracker_script),
        "--video",   str(cfg.video_path),
        "--output",  str(crop_map_path),
        "--sample-rate",   str(cfg.tracker_sample_rate),
        "--smooth-window", str(cfg.tracker_smooth_window),
    ]

    log.info("CMD: %s", " ".join(cmd))
    r = subprocess.run(cmd)   # hereda stdout/stderr para ver el progreso

    if r.returncode != 0:
        log.error("face_tracker.py falló (código %d).", r.returncode)
        log.warning("Continuando sin face tracking (crop centrado).")
        return None

    if not crop_map_path.exists():
        log.error("face_tracker.py terminó pero no generó el archivo.")
        return None

    with open(crop_map_path, encoding="utf-8") as f:
        return json.load(f)


def get_crop_x_at(crop_map: dict, t_sec: float) -> int:
    """
    Interpolación lineal del crop_x para un tiempo dado.
    Importado conceptualmente de face_tracker.py para mantener el módulo independiente.
    """
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

    alpha = (t_sec - t0) / (t1 - t0)
    return int(x0 + alpha * (x1 - x0))


# ──────────────────────────────────────────────
# CROP DINÁMICO — GENERACIÓN DE FILTRO
# ──────────────────────────────────────────────

def compute_crop_width(src_w: int, src_h: int) -> int:
    crop_w = math.floor(src_h * 9 / 16)
    if crop_w > src_w:
        crop_w = src_w
    return crop_w


def build_dynamic_crop_filter(
    crop_map:   dict,
    start_sec:  float,
    end_sec:    float,
    src_w:      int,
    src_h:      int,
    fps:        float,
) -> str:
    """
    Genera una expresión de crop dinámico para ffmpeg usando la función crop
    con expresión de tiempo 't'.

    Estrategia: genera una expresión ffmpeg usando 'if(between(t,...))' para
    cada intervalo de 1 segundo del clip. Es verboso pero garantiza fluidez
    porque usa interpolación dentro de cada intervalo.

    Para clips cortos (<= 180s) esto es manejable (~180 expresiones).
    Para clips más largos, reduce la densidad automáticamente.
    """
    crop_w = crop_map["crop_width"]
    crop_h = src_h
    if crop_w > src_w:
        crop_w = src_w
        crop_h = math.floor(src_w * 16 / 9)

    crop_y = (src_h - crop_h) // 2
    duration = end_sec - start_sec

    # Muestrear la posición X cada 0.5s dentro del clip
    # Usa interpolación lineal entre muestras para suavidad
    sample_interval = 0.5
    n_samples = max(2, int(duration / sample_interval) + 1)
    times  = [start_sec + i * (duration / (n_samples - 1)) for i in range(n_samples)]
    crop_xs = [get_crop_x_at(crop_map, t) for t in times]

    # Construir expresión ffmpeg con lerp entre keyframes
    # t_rel = tiempo relativo al inicio del clip (PTS-STARTPTS ya aplicado antes del crop)
    # ffmpeg permite expresiones con 'if', 'between', operaciones aritméticas
    #
    # Forma: construir expresión piecewise linear con 'if(lte(t, T1), lerp, rest)'
    # Para N keyframes: if(lte(t,t1), x0+(x1-x0)*(t-t0)/(t1-t0), if(lte(t,t2), ..., xN))

    # Tiempos relativos al clip
    rel_times = [t - start_sec for t in times]

    # Construir expresión anidada (de adentro hacia afuera)
    # El último segmento es el valor final
    expr = str(crop_xs[-1])

    for i in range(n_samples - 2, -1, -1):
        t0_r = rel_times[i]
        t1_r = rel_times[i + 1]
        x0   = crop_xs[i]
        x1   = crop_xs[i + 1]
        dt   = t1_r - t0_r

        if dt < 0.001:
            lerp = str(x0)
        else:
            # lerp = x0 + (x1-x0) * (t - t0) / dt
            dx = x1 - x0
            if dx == 0:
                lerp = str(x0)
            else:
                sign = "+" if dx > 0 else "-"
                lerp = f"{x0}{sign}{abs(dx)}*(t-{t0_r:.3f})/{dt:.3f}"

        expr = f"if(lte(t,{t1_r:.3f}),{lerp},{expr})"

    # Clampear al rango válido
    max_x = src_w - crop_w
    expr_clamped = f"clip({expr},0,{max_x})"

    return f"crop={crop_w}:{crop_h}:{expr_clamped}:{crop_y}"


def build_static_crop_filter(src_w: int, src_h: int, offset_x: int = 0) -> str:
    """Crop fijo centrado (fallback cuando no hay tracking)."""
    crop_w = math.floor(src_h * 9 / 16)
    if crop_w > src_w:
        crop_w = src_w
        crop_h = math.floor(src_w * 16 / 9)
    else:
        crop_h = src_h

    crop_x = max(0, (src_w - crop_w) // 2 + offset_x)
    crop_x = min(crop_x, src_w - crop_w)
    crop_y = (src_h - crop_h) // 2

    return f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"


# ──────────────────────────────────────────────
# ASS SUBTITLES
# ──────────────────────────────────────────────

def seconds_to_ass(t: float) -> str:
    h  = int(t // 3600)
    m  = int((t % 3600) // 60)
    s  = int(t % 60)
    cs = int(round((t % 1) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def wrap_for_ass(text: str, max_chars: int) -> str:
    words = text.split()
    if not words or len(text) <= max_chars:
        return text

    lines:   list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        wlen = len(word)
        if current and current_len + 1 + wlen > max_chars:
            lines.append(" ".join(current))
            current     = [word]
            current_len = wlen
        else:
            current_len += (1 + wlen) if current_len > 0 else wlen
            current.append(word)

    if current:
        lines.append(" ".join(current))

    return r"\N".join(lines)


def generate_ass(segments: list[dict], output_path: Path, cfg: Config) -> None:
    margin_v    = int(cfg.height * (1.0 - cfg.subtitle_y_pct))
    col_white   = "&H00FFFFFF"
    col_black   = "&H00000000"
    col_shadow  = "&H88000000"

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {cfg.width}\n"
        f"PlayResY: {cfg.height}\n"
        "ScaledBorderAndShadow: yes\n"
        "WrapStyle: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Arial,{cfg.font_size},"
        f"{col_white},&H000000FF,{col_black},{col_shadow},"
        f"-1,0,0,0,100,100,0,0,1,4,2,2,60,60,{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    for seg in segments:
        t_start = seconds_to_ass(seg["start"])
        t_end   = seconds_to_ass(seg["end"])
        text    = wrap_for_ass(seg["text"], cfg.max_chars_per_line)
        if text:
            text = text[0].upper() + text[1:]
        lines.append(f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{text}")

    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))
        f.write("\n")


# ──────────────────────────────────────────────
# FFMPEG COMMAND BUILDER
# ──────────────────────────────────────────────

def build_ffmpeg_cmd(
    video_path:   Path,
    output_path:  Path,
    ass_path:     Path,
    start_sec:    float,
    end_sec:      float,
    crop_filter:  str,
    cfg:          Config,
) -> list[str]:
    duration  = end_sec - start_sec
    ass_str   = str(ass_path.resolve())

    vf = (
        f"trim=start={start_sec}:end={end_sec},"
        f"setpts=PTS-STARTPTS,"
        f"{crop_filter},"
        f"scale={cfg.width}:{cfg.height}:flags=lanczos,"
        f"subtitles='{ass_str}'"
    )
    af = f"atrim=start={start_sec}:end={end_sec},asetpts=PTS-STARTPTS"

    return [
        "ffmpeg", "-y",
        "-i",        str(video_path),
        "-vf",       vf,
        "-af",       af,
        "-c:v",      "libx264",
        "-preset",   "fast",
        "-crf",      "22",
        "-c:a",      "aac",
        "-b:a",      "192k",
        "-movflags", "+faststart",
        "-t",        str(duration),
        str(output_path),
    ]


def run_ffmpeg(cmd: list[str], label: str) -> bool:
    log.info("Procesando: %s", label)
    log.debug("CMD: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("ffmpeg falló para '%s':\n%s", label, r.stderr[-3000:])
        return False
    log.info("✓ OK: %s", label)
    return True


def concat_clips(clip_paths: list[Path], output_path: Path) -> bool:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        concat_file = f.name
        for p in clip_paths:
            f.write(f"file '{p.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(concat_file)

    if r.returncode != 0:
        log.error("Concat falló:\n%s", r.stderr[-2000:])
        return False

    log.info("✓ Compilado: %s", output_path)
    return True


# ──────────────────────────────────────────────
# ARG PARSER
# ──────────────────────────────────────────────

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Genera clips 9:16 con face tracking y subtítulos ASS."
    )
    p.add_argument("--predica",   required=True)
    p.add_argument("--whisper",   required=True,
                   help="Archivo .json o directorio de transcripción Whisper")
    p.add_argument("--video",     required=True)
    p.add_argument("--output",    default="./output")
    p.add_argument("--resolution", default="1080x1920")
    p.add_argument("--no-concat", action="store_true")
    p.add_argument("--font-size", type=int,   default=72)
    p.add_argument("--subtitle-y", type=float, default=0.72)
    p.add_argument("--max-chars",  type=int,   default=28)
    # Tracking
    p.add_argument("--no-tracking", action="store_true",
                   help="Deshabilitar face tracking (usar crop centrado)")
    p.add_argument("--crop-map", default=None,
                   help="Ruta custom al crop_map.json (omitir = auto)")
    p.add_argument("--tracker-sample-rate",   type=int, default=1,
                   help="FPS de análisis del tracker (default: 1)")
    p.add_argument("--tracker-smooth-window", type=int, default=3,
                   help="Segundos de suavizado del tracker (default: 3)")
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    w, h = args.resolution.split("x")

    return Config(
        predica_json           = Path(args.predica),
        whisper_path           = Path(args.whisper),
        video_path             = Path(args.video),
        output_dir             = Path(args.output),
        width                  = int(w),
        height                 = int(h),
        concat                 = not args.no_concat,
        font_size              = args.font_size,
        subtitle_y_pct         = args.subtitle_y,
        max_chars_per_line     = args.max_chars,
        use_tracking           = not args.no_tracking,
        crop_map_path          = Path(args.crop_map) if args.crop_map else None,
        tracker_sample_rate    = args.tracker_sample_rate,
        tracker_smooth_window  = args.tracker_smooth_window,
    )


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main() -> None:
    cfg = parse_args()

    check_ffmpeg()

    for path, label in [(cfg.predica_json, "--predica"), (cfg.video_path, "--video")]:
        if not path.exists():
            log.error("No encontrado (%s): %s", label, path)
            sys.exit(1)

    whisper_file = resolve_whisper_path(cfg.whisper_path, cfg.predica_json)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ass_dir = cfg.output_dir / "_ass_tmp"
    ass_dir.mkdir(exist_ok=True)

    # ── Cargar datos ──
    predica      = load_predica_json(cfg.predica_json)
    all_segments = load_whisper_segments(whisper_file)
    key_moments  = predica.get("key_moments", [])

    if not key_moments:
        log.error("No se encontraron key_moments.")
        sys.exit(1)

    src_w, src_h = get_video_dimensions(cfg.video_path)

    # ── Face tracking ──
    crop_map = load_or_generate_crop_map(cfg)

    if crop_map:
        log.info("Face tracking: ACTIVO (crop dinámico)")
        fps = get_video_fps(cfg.video_path)
    else:
        log.info("Face tracking: INACTIVO (crop centrado fijo)")
        fps = None

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("Video:       %s  (%dx%d)", cfg.video_path.name, src_w, src_h)
    log.info("Salida:      %dx%d", cfg.width, cfg.height)
    log.info("Key moments: %d", len(key_moments))
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    generated_clips: list[Path] = []

    for km in key_moments:
        km_id  = km["id"]
        title  = km["title"]
        start  = float(km["start_seconds"])
        end    = float(km["end_seconds"])

        safe_title = re.sub(r"[^\w\s\-]", "_", title).strip().replace(" ", "_")[:50]
        clip_path  = cfg.output_dir / f"{km_id:02d}_{safe_title}.mp4"
        ass_path   = ass_dir        / f"{km_id:02d}_{safe_title}.ass"

        log.info(
            "── [%d/%d] %s  [%.0fs → %.0fs]",
            km_id, len(key_moments), title, start, end
        )

        # Subtítulos
        clip_segs = segments_for_range(all_segments, start, end)
        log.info("   Segmentos Whisper: %d", len(clip_segs))
        generate_ass(clip_segs, ass_path, cfg)

        # Crop filter
        if crop_map:
            crop_filter = build_dynamic_crop_filter(
                crop_map, start, end, src_w, src_h, fps
            )
            log.debug("   Crop dinámico generado")
        else:
            crop_filter = build_static_crop_filter(src_w, src_h)
            log.debug("   Crop estático centrado")

        cmd = build_ffmpeg_cmd(
            video_path  = cfg.video_path,
            output_path = clip_path,
            ass_path    = ass_path,
            start_sec   = start,
            end_sec     = end,
            crop_filter = crop_filter,
            cfg         = cfg,
        )

        if run_ffmpeg(cmd, f"{km_id:02d}_{safe_title}"):
            generated_clips.append(clip_path)
        else:
            log.warning("   ✗ Clip %d omitido.", km_id)

    # Limpiar ASS temporales
    for f in ass_dir.glob("*.ass"):
        f.unlink()
    try:
        ass_dir.rmdir()
    except OSError:
        pass

    if cfg.concat and len(generated_clips) > 1:
        concat_path = cfg.output_dir / "00_compilado_completo.mp4"
        log.info("Compilado (%d clips)...", len(generated_clips))
        concat_clips(generated_clips, concat_path)

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("Clips generados: %d / %d", len(generated_clips), len(key_moments))
    log.info("Salida: %s", cfg.output_dir.resolve())
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()

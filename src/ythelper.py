# ==========================
# USO / REFERENCIA RÁPIDA
# ==========================
#
# Modos disponibles:
#
#   download-audio  → Descarga solo audio WAV a source/
#   download-video  → Descarga solo video (máx calidad) a videos/
#   download-all    → Descarga audio WAV + video en una sola pasada
#   transcribe      → Transcribe los .wav/.mp3 que haya en source/
#   webhook         → Envía los .json de transcripts/ al webhook (lotes de 3)
#   run-audio       → Pipeline: download-audio → transcribe → webhook
#   run-full        → Pipeline: download-all → transcribe → webhook
#
# Ejemplos:
#
#   python ythelper.py --mode download-audio --ids abc123 def456
#   python ythelper.py --mode download-audio --ids-file ids.txt
#   python ythelper.py --mode download-video --ids abc123
#   python ythelper.py --mode download-all   --ids abc123
#   python ythelper.py --mode transcribe
#   python ythelper.py --mode webhook
#   python ythelper.py --mode run-audio --ids abc123
#   python ythelper.py --mode run-full  --ids-file ids.txt
#
# Modo resume (reintenta los IDs que fallaron o quedaron pendientes):
#
#   python ythelper.py --mode run-full --resume
#   python ythelper.py --mode run-audio --resume
#
#   --resume lee logs/progress.json y saltea los IDs con status DONE.
#   No requiere --ids ni --ids-file (los toma del progress.json).
#
# Archivo de IDs (--ids-file):
#   - Un ID por línea (recomendado)
#   - O varios IDs separados por espacio en la misma línea
#   - Las líneas que empiezan con # se ignoran (sirven de comentario)
#   - Ejemplo de ids.txt:
#       # tanda del lunes
#       k-HJKYrnQPQ
#       5YguWbAXBmc EKSOT7A7ghc
#       # tanda del martes
#       1kwue3QCOyo
#
# Variables de entorno requeridas en .env:
#   WEBHOOK_URL      → URL del webhook destino
#   WEBHOOK_SECRET   → Header de autenticación (X-Webhook-Secret)
#   YT_COOKIES_PATH  → Ruta al archivo de cookies de YouTube
#
# Estructura de carpetas:
#   source/      → Audios WAV listos para transcribir
#   videos/      → Videos descargados (máxima calidad)
#   processed/   → Audios ya transcriptos (movidos desde source/)
#   transcripts/      → JSONs pendientes de enviar al webhook
#   transcripts/done/ → JSONs enviados exitosamente (webhook 200)
#   logs/        → Logs por ejecución (fecha.log) + progress.json
#
# ==========================

import os
import json
import shutil
import subprocess
import requests
import argparse
import time
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ==========================
# CONFIGURACIÓN
# ==========================

SOURCE      = "source"
VIDEOS      = "videos"
PROCESSED   = "processed"
TRANSCRIPTS      = "transcripts"
TRANSCRIPTS_DONE = "transcripts/done"
LOGS_DIR         = "logs"
PROGRESS_FILE = os.path.join(LOGS_DIR, "progress.json")

WEBHOOK_URL    = os.getenv("WEBHOOK_URL")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
COOKIES_PATH   = os.getenv("YT_COOKIES_PATH")
MODEL_SIZE   = "small"

FILENAME_TEMPLATE = "%(upload_date)s_%(id)s_%(title)s.%(ext)s"


# ==========================
# LOGGING
# ==========================

def setup_logging():
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(LOGS_DIR, f"{timestamp}.log")

    fmt = "[%(asctime)s] %(levelname)-5s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logging.info(f"📋 Log: {log_file}")
    return log_file

log = logging.getLogger(__name__)


# ==========================
# PROGRESS TRACKER
# ==========================

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=4, ensure_ascii=False)


def update_progress(progress, video_id, status, step=None, error=None):
    entry = {"status": status}
    if step:
        entry["step"] = step
    if error:
        entry["error"] = str(error)
    entry["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress[video_id] = entry
    save_progress(progress)


# ==========================
# HELPERS
# ==========================

def check_cookies():
    if not COOKIES_PATH:
        log.error("YT_COOKIES_PATH no está definida en el .env")
        sys.exit(1)


def load_ids(args):
    ids = []

    if args.ids:
        ids.extend(args.ids)

    if args.ids_file:
        if not os.path.exists(args.ids_file):
            log.error(f"Archivo no encontrado: {args.ids_file}")
            sys.exit(1)
        with open(args.ids_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.extend(line.split())

    if not ids:
        log.error("No se proporcionaron IDs. Usá --ids o --ids-file.")
        sys.exit(1)

    return ids


def load_ids_for_resume():
    """Carga todos los IDs del progress.json que no están DONE."""
    progress = load_progress()
    if not progress:
        log.error("No existe progress.json. Ejecutá primero sin --resume.")
        sys.exit(1)

    pending = [vid_id for vid_id, data in progress.items() if data.get("status") != "DONE"]
    if not pending:
        log.info("✅ Todos los IDs en progress.json ya están DONE. Nada que retomar.")
        sys.exit(0)

    log.info(f"🔁 Resume: {len(pending)} IDs pendientes o fallidos de {len(progress)} totales")
    return pending, progress


def yt_url(video_id):
    return f"https://www.youtube.com/watch?v={video_id}"


def print_summary(progress):
    done    = sum(1 for d in progress.values() if d.get("status") == "DONE")
    failed  = sum(1 for d in progress.values() if d.get("status") == "FAILED")
    total   = len(progress)
    log.info(f"")
    log.info(f"{'='*45}")
    log.info(f"  RESUMEN FINAL")
    log.info(f"  Total  : {total}")
    log.info(f"  ✅ OK   : {done}")
    log.info(f"  ❌ Falló: {failed}")
    log.info(f"{'='*45}")
    if failed:
        failed_ids = [vid for vid, d in progress.items() if d.get("status") == "FAILED"]
        for vid in failed_ids:
            step  = progress[vid].get("step", "?")
            error = progress[vid].get("error", "?")
            log.info(f"  ❌ {vid} | step: {step} | error: {error}")
        log.info(f"  → Podés reintentar con --resume")
    log.info(f"{'='*45}")


# ==========================
# DESCARGA
# ==========================

def _run_yt_dlp(cmd, video_id, step):
    """Ejecuta yt-dlp y retorna True si tuvo éxito."""
    try:
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode == 0:
            return True
        else:
            log.error(f"yt-dlp terminó con código {result.returncode} | {video_id} | {step}")
            return False
    except Exception as e:
        log.error(f"Excepción al ejecutar yt-dlp | {video_id} | {step} | {e}")
        return False


def download_audio_for_id(video_id):
    cmd = [
        "yt-dlp",
        "--cookies", COOKIES_PATH,
        yt_url(video_id),
        "-f", "bv*+ba/b",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--no-playlist",
        "-o", f"{SOURCE}/{FILENAME_TEMPLATE}"
    ]
    return _run_yt_dlp(cmd, video_id, "download-audio")


def download_video_for_id(video_id):
    cmd = [
        "yt-dlp",
        "--cookies", COOKIES_PATH,
        yt_url(video_id),
        "-f", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", f"{VIDEOS}/{FILENAME_TEMPLATE}"
    ]
    return _run_yt_dlp(cmd, video_id, "download-video")


def download_audio(ids, progress):
    os.makedirs(SOURCE, exist_ok=True)
    check_cookies()
    total = len(ids)
    log.info(f"🎵 Descargando audio de {total} video(s)...")

    for i, video_id in enumerate(ids, 1):
        log.info(f"⬇ [{i}/{total}] Audio → {video_id}")
        update_progress(progress, video_id, "IN_PROGRESS", step="download-audio")

        ok = download_audio_for_id(video_id)
        if ok:
            log.info(f"✅ [{i}/{total}] Audio OK → {video_id}")
            update_progress(progress, video_id, "DONE")
        else:
            log.error(f"❌ [{i}/{total}] Audio FAILED → {video_id}")
            update_progress(progress, video_id, "FAILED", step="download-audio", error="yt-dlp error")


def download_video(ids, progress):
    os.makedirs(VIDEOS, exist_ok=True)
    check_cookies()
    total = len(ids)
    log.info(f"🎬 Descargando video de {total} video(s)...")

    for i, video_id in enumerate(ids, 1):
        log.info(f"⬇ [{i}/{total}] Video → {video_id}")
        update_progress(progress, video_id, "IN_PROGRESS", step="download-video")

        ok = download_video_for_id(video_id)
        if ok:
            log.info(f"✅ [{i}/{total}] Video OK → {video_id}")
            update_progress(progress, video_id, "DONE")
        else:
            log.error(f"❌ [{i}/{total}] Video FAILED → {video_id}")
            update_progress(progress, video_id, "FAILED", step="download-video", error="yt-dlp error")


def download_audio_and_video(ids, progress):
    os.makedirs(SOURCE, exist_ok=True)
    os.makedirs(VIDEOS, exist_ok=True)
    check_cookies()
    total = len(ids)
    log.info(f"🎬🎵 Descargando audio + video de {total} video(s)...")

    for i, video_id in enumerate(ids, 1):
        log.info(f"⬇ [{i}/{total}] Video → {video_id}")
        update_progress(progress, video_id, "IN_PROGRESS", step="download-video")
        ok_video = download_video_for_id(video_id)

        if ok_video:
            log.info(f"✅ [{i}/{total}] Video OK → {video_id}")
        else:
            log.error(f"❌ [{i}/{total}] Video FAILED → {video_id}")
            update_progress(progress, video_id, "FAILED", step="download-video", error="yt-dlp error")

        log.info(f"⬇ [{i}/{total}] Audio → {video_id}")
        update_progress(progress, video_id, "IN_PROGRESS", step="download-audio")
        ok_audio = download_audio_for_id(video_id)

        if ok_audio:
            log.info(f"✅ [{i}/{total}] Audio OK → {video_id}")
            # Solo marca DONE si ambos tuvieron éxito
            if ok_video:
                update_progress(progress, video_id, "DONE", step="download-all")
            else:
                update_progress(progress, video_id, "FAILED", step="download-video", error="video failed, audio ok")
        else:
            log.error(f"❌ [{i}/{total}] Audio FAILED → {video_id}")
            update_progress(progress, video_id, "FAILED", step="download-audio", error="yt-dlp error")


# ==========================
# TRANSCRIPCIÓN (WHISPER)
# ==========================

def get_clean_video_id(filename):
    """Extrae el ID de 11 caracteres después del primer guion bajo."""
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_', 1)
    if len(parts) >= 2:
        return parts[1][:11]
    return base_name


def process_whisper_file(filename, model, progress, index, total):
    base_name = os.path.splitext(filename)[0]
    video_id  = get_clean_video_id(filename)
    filepath  = os.path.join(SOURCE, filename)
    json_path = os.path.join(TRANSCRIPTS, f"{base_name}.json")

    log.info(f"🎧 [{index}/{total}] Transcribiendo → {filename} (ID: {video_id})")
    update_progress(progress, video_id, "IN_PROGRESS", step="transcribe")

    try:
        segments, info = model.transcribe(filepath, language="es")

        result = {
            "videoId":  video_id,
            "language": info.language,
            "duration": info.duration,
            "segments": []
        }

        for segment in segments:
            percent = (segment.end / info.duration) * 100
            sys.stdout.write(f"\r   ⏳ Progreso: {percent:.2f}% | {segment.end:.1f}/{info.duration:.1f} seg")
            sys.stdout.flush()

            result["segments"].append({
                "start": segment.start,
                "end":   segment.end,
                "text":  segment.text.strip()
            })

        print()  # salto de línea tras la barra de progreso
        log.info(f"   💾 Guardando transcripción → {json_path}")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        shutil.move(filepath, os.path.join(PROCESSED, filename))
        update_progress(progress, video_id, "DONE", step="transcribe")
        return json_path

    except Exception as e:
        log.error(f"❌ [{index}/{total}] Transcripción FAILED → {filename} | {e}")
        update_progress(progress, video_id, "FAILED", step="transcribe", error=e)
        return None


def load_whisper_model():
    log.info(f"🚀 Cargando modelo Whisper ({MODEL_SIZE})...")
    from faster_whisper import WhisperModel
    return WhisperModel(MODEL_SIZE, compute_type="int8", cpu_threads=4)


def run_transcribe(progress):
    model = load_whisper_model()
    files = sorted([f for f in os.listdir(SOURCE) if f.lower().endswith(('.wav', '.mp3'))])
    total = len(files)
    log.info(f"📂 Archivos a transcribir: {total}")
    json_paths = []
    for i, f in enumerate(files, 1):
        json_path = process_whisper_file(f, model, progress, i, total)
        if json_path:
            json_paths.append(json_path)
    return model, files, json_paths


# ==========================
# WEBHOOK
# ==========================

def send_webhook(json_path, progress):
    filename = os.path.basename(json_path)
    video_id = get_clean_video_id(filename)
    log.info(f"🌐 Webhook → {filename}")
    update_progress(progress, video_id, "IN_PROGRESS", step="webhook")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        headers = {"X-Webhook-Secret": WEBHOOK_SECRET} if WEBHOOK_SECRET else {}
        r = requests.post(WEBHOOK_URL, json=data, headers=headers, timeout=30)
        if r.status_code == 200:
            log.info(f"   ✅ Webhook OK → status {r.status_code}")
            shutil.move(json_path, os.path.join(TRANSCRIPTS_DONE, filename))
            log.info(f"   📁 Movido → transcripts/done/{filename}")
            update_progress(progress, video_id, "DONE", step="webhook")
        else:
            log.error(f"   ❌ Webhook respondió {r.status_code} → {filename}")
            update_progress(progress, video_id, "FAILED", step="webhook", error=f"status {r.status_code}")
    except Exception as e:
        log.error(f"   ❌ Webhook FAILED → {filename} | {e}")
        update_progress(progress, video_id, "FAILED", step="webhook", error=e)


def run_webhook(progress):
    files = sorted([f for f in os.listdir(TRANSCRIPTS) if f.endswith('.json')])
    total = len(files)
    log.info(f"🚀 Modo Webhook | {total} archivos")

    for i in range(0, total, 3):
        batch = files[i:i+3]
        log.info(f"📦 Lote {i//3 + 1}: {batch}")
        for f in batch:
            send_webhook(os.path.join(TRANSCRIPTS, f), progress)

        if i + 3 < total:
            log.info(f"⏳ Cooldown: esperando 2 minutos...")
            time.sleep(120)


# ==========================
# MAIN
# ==========================

def main():
    parser = argparse.ArgumentParser(description="ythelper: YouTube → Whisper → Webhook")
    parser.add_argument(
        "--mode",
        choices=[
            "download-audio",
            "download-video",
            "download-all",
            "transcribe",
            "webhook",
            "run-audio",
            "run-full",
        ],
        required=True,
        help="Modo de ejecución"
    )
    parser.add_argument("--ids",      nargs="+", help="IDs de YouTube")
    parser.add_argument("--ids-file", help="Archivo .txt con IDs de YouTube")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retomar desde progress.json, salteando los IDs ya DONE"
    )
    args = parser.parse_args()

    # Crear carpetas base
    for d in [SOURCE, VIDEOS, PROCESSED, TRANSCRIPTS, TRANSCRIPTS_DONE, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

    setup_logging()

    log.info(f"{'='*45}")
    log.info(f"  ythelper | modo: {args.mode} | resume: {args.resume}")
    log.info(f"{'='*45}")

    # Cargar o inicializar progress
    progress = load_progress()

    # Resolver IDs según --resume o argumentos
    needs_ids = args.mode in ("download-audio", "download-video", "download-all", "run-audio", "run-full")

    if needs_ids:
        if args.resume:
            ids, progress = load_ids_for_resume()
        else:
            ids = load_ids(args)
            # Registrar todos los IDs como PENDING al inicio
            for vid in ids:
                if vid not in progress:
                    update_progress(progress, vid, "PENDING")
        total_ids = len(ids)
        log.info(f"📋 IDs a procesar: {total_ids}")

    # ── DOWNLOAD-AUDIO ────────────────────────────────────────
    if args.mode == "download-audio":
        download_audio(ids, progress)

    # ── DOWNLOAD-VIDEO ────────────────────────────────────────
    elif args.mode == "download-video":
        download_video(ids, progress)

    # ── DOWNLOAD-ALL ──────────────────────────────────────────
    elif args.mode == "download-all":
        download_audio_and_video(ids, progress)

    # ── TRANSCRIBE ────────────────────────────────────────────
    elif args.mode == "transcribe":
        run_transcribe(progress)

    # ── WEBHOOK ───────────────────────────────────────────────
    elif args.mode == "webhook":
        run_webhook(progress)

    # ── RUN-AUDIO ─────────────────────────────────────────────
    elif args.mode == "run-audio":
        log.info("▶ Paso 1/3: Descarga de audio")
        download_audio(ids, progress)

        log.info("▶ Paso 2/3: Transcripción")
        model = load_whisper_model()
        files = sorted([f for f in os.listdir(SOURCE) if f.lower().endswith(('.wav', '.mp3'))])
        total = len(files)
        for i, f in enumerate(files, 1):
            json_path = process_whisper_file(f, model, progress, i, total)
            if json_path:
                log.info("▶ Paso 3/3: Webhook")
                send_webhook(json_path, progress)

    # ── RUN-FULL ──────────────────────────────────────────────
    elif args.mode == "run-full":
        log.info("▶ Paso 1/3: Descarga de audio + video")
        download_audio_and_video(ids, progress)

        log.info("▶ Paso 2/3: Transcripción")
        model = load_whisper_model()
        files = sorted([f for f in os.listdir(SOURCE) if f.lower().endswith(('.wav', '.mp3'))])
        total = len(files)
        for i, f in enumerate(files, 1):
            json_path = process_whisper_file(f, model, progress, i, total)
            if json_path:
                log.info("▶ Paso 3/3: Webhook")
                send_webhook(json_path, progress)

    print_summary(progress)
    log.info("✨ ¡Todo terminado!")


if __name__ == "__main__":
    main()

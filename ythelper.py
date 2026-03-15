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
#                     Con --file envía solo ese JSON específico (sin lotes ni cooldown)
#   run-audio       → Pipeline intercalado por video: descarga → transcribe → webhook
#   run-full        → Pipeline intercalado por video: descarga audio+video → transcribe → webhook
#   local           → Pipeline desde archivo local (NAS): transcribe → webhook
#                     El video original se mueve a processed/ dentro del NAS al terminar
#
# Ejemplos:
#
#   python ythelper.py --mode download-audio --ids abc123 def456
#   python ythelper.py --mode download-audio --ids-file ids.txt
#   python ythelper.py --mode download-video --ids abc123
#   python ythelper.py --mode download-all   --ids abc123
#   python ythelper.py --mode transcribe
#   python ythelper.py --mode webhook
#   python ythelper.py --mode webhook --file /ruta/al/transcript.json
#   python ythelper.py --mode webhook --file t1.json t2.json
#   python ythelper.py --mode run-audio --ids abc123
#   python ythelper.py --mode run-full  --ids-file ids.txt
#   python ythelper.py --mode local --file /mnt/nas/sermones/video.mp4
#   python ythelper.py --mode local --local-folder /mnt/nas/sermones/
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
#   WEBHOOK_URL      → URL del webhook destino (producción)
#   WEBHOOK_URL_TEST → URL del webhook destino (pruebas, usar con --test)
#   WEBHOOK_SECRET   → Header de autenticación (X-Webhook-Secret)
#   YT_COOKIES_PATH  → Ruta al archivo de cookies de YouTube
#
# Estructura de carpetas:
#   data/input/source/           → Audios WAV listos para transcribir
#   data/input/videos/           → Videos descargados (máxima calidad)
#   data/output/processed/       → Audios ya transcriptos (movidos desde source/)
#   data/output/transcripts/     → JSONs pendientes de enviar al webhook
#   data/output/transcripts/done/→ JSONs enviados exitosamente (webhook 200)
#   data/logs/                   → Logs por ejecución (fecha.log) + progress.json
#
# Comportamiento de auto-retry en run-audio / run-full:
#   - El pipeline procesa cada video de forma intercalada:
#     descarga → transcribe → webhook → siguiente video
#   - La transcripción actúa como cooldown natural entre descargas de YT
#   - Al finalizar el loop principal, si quedan IDs con FAILED en
#     step download-audio o download-video, se ejecutan hasta 2 rondas
#     de reintento con 5 minutos de pausa entre cada intento individual
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

SOURCE           = "data/input/source"
VIDEOS           = "data/input/videos"
PROCESSED        = "data/output/processed"
TRANSCRIPTS      = "data/output/transcripts"
TRANSCRIPTS_DONE = "data/output/transcripts/done"
LOGS_DIR         = "data/logs"
PROGRESS_FILE    = os.path.join(LOGS_DIR, "progress.json")

WEBHOOK_URL      = os.getenv("WEBHOOK_URL")
WEBHOOK_URL_TEST = os.getenv("WEBHOOK_URL_TEST")
WEBHOOK_SECRET   = os.getenv("WEBHOOK_SECRET")
COOKIES_PATH     = os.getenv("YT_COOKIES_PATH")
MODEL_SIZE       = "small"

# Se sobreescribe en main() según --test
ACTIVE_WEBHOOK_URL = None

FILENAME_TEMPLATE  = "%(upload_date)s_%(id)s_%(title)s.%(ext)s"
RETRY_MAX_ATTEMPTS = 2
RETRY_WAIT_SECONDS = 300  # 5 minutos


# ==========================
# LOGGING
# ==========================

def setup_logging():
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(LOGS_DIR, f"{timestamp}.log")

    fmt     = "[%(asctime)s] %(levelname)-5s %(message)s"
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


def find_downloaded_audio(video_id):
    """Busca en source/ el archivo WAV que corresponde a un video_id."""
    for f in os.listdir(SOURCE):
        if video_id in f and f.lower().endswith(".wav"):
            return f
    return None


def print_summary(progress):
    done   = sum(1 for d in progress.values() if d.get("status") == "DONE")
    failed = sum(1 for d in progress.values() if d.get("status") == "FAILED")
    total  = len(progress)
    log.info("")
    log.info(f"{'='*45}")
    log.info(f"  RESUMEN FINAL")
    log.info(f"  Total   : {total}")
    log.info(f"  ✅ OK    : {done}")
    log.info(f"  ❌ Falló : {failed}")
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
    """Descarga solo audio para una lista de IDs (modo standalone)."""
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
    """Descarga solo video para una lista de IDs (modo standalone)."""
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
    """Descarga audio + video para una lista de IDs (modo standalone)."""
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
            if ok_video:
                update_progress(progress, video_id, "DONE", step="download-all")
            else:
                update_progress(progress, video_id, "FAILED", step="download-video", error="video failed, audio ok")
        else:
            log.error(f"❌ [{i}/{total}] Audio FAILED → {video_id}")
            update_progress(progress, video_id, "FAILED", step="download-audio", error="yt-dlp error")


# ==========================
# AUTO-RETRY DESCARGAS
# ==========================

DOWNLOAD_FAIL_STEPS = ("download-audio", "download-video", "download-all")


def get_failed_downloads(progress, ids):
    """Retorna los IDs del batch actual que fallaron en algún paso de descarga."""
    return [
        vid for vid in ids
        if progress.get(vid, {}).get("status") == "FAILED"
        and progress.get(vid, {}).get("step") in DOWNLOAD_FAIL_STEPS
    ]


def auto_retry_downloads(ids, progress, include_video=False):
    """
    Reintenta la descarga de audio (y video si include_video=True)
    para los IDs fallidos del batch.
    Máximo RETRY_MAX_ATTEMPTS rondas con RETRY_WAIT_SECONDS de pausa
    entre cada intento individual dentro de una ronda.
    """
    failed = get_failed_downloads(progress, ids)

    if not failed:
        return

    log.info("")
    log.info(f"🔄 Auto-retry: {len(failed)} IDs fallidos en descarga")

    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        if not failed:
            break

        log.info(f"{'='*45}")
        log.info(f"  AUTO-RETRY ronda {attempt}/{RETRY_MAX_ATTEMPTS} | {len(failed)} ID(s)")
        log.info(f"{'='*45}")

        still_failed = []

        for i, video_id in enumerate(failed, 1):
            # Pausa antes de cada intento (excepto el primero de la ronda)
            if i > 1:
                log.info(f"⏳ Esperando {RETRY_WAIT_SECONDS // 60} min antes del siguiente reintento...")
                time.sleep(RETRY_WAIT_SECONDS)

            log.info(f"🔄 [{i}/{len(failed)}] Reintentando → {video_id} (ronda {attempt})")

            ok_video = True  # asumir OK si no se necesita video
            ok_audio = False

            if include_video:
                update_progress(progress, video_id, "IN_PROGRESS", step="download-video")
                ok_video = download_video_for_id(video_id)
                if ok_video:
                    log.info(f"✅ Video OK → {video_id}")
                else:
                    log.error(f"❌ Video FAILED → {video_id}")
                    update_progress(progress, video_id, "FAILED", step="download-video",
                                    error=f"retry {attempt} failed")

            update_progress(progress, video_id, "IN_PROGRESS", step="download-audio")
            ok_audio = download_audio_for_id(video_id)

            if ok_audio and ok_video:
                log.info(f"✅ Reintento exitoso → {video_id}")
                update_progress(progress, video_id, "DONE", step="download-audio")

                # Continuar con transcripción y webhook para este ID recuperado
                audio_file = find_downloaded_audio(video_id)
                if audio_file:
                    from faster_whisper import WhisperModel
                    log.info(f"🚀 Cargando modelo Whisper para reintento...")
                    model = WhisperModel(MODEL_SIZE, compute_type="int8", cpu_threads=4)
                    json_path = process_whisper_file(audio_file, model, progress, i, len(failed))
                    if json_path:
                        send_webhook(json_path, progress)
            else:
                log.error(f"❌ Reintento fallido → {video_id} (ronda {attempt})")
                failed_step = "download-audio" if not ok_audio else "download-video"
                update_progress(progress, video_id, "FAILED", step=failed_step,
                                error=f"retry {attempt} failed")
                still_failed.append(video_id)

        failed = still_failed

    if failed:
        log.error(f"❌ {len(failed)} ID(s) siguen fallando tras {RETRY_MAX_ATTEMPTS} reintentos: {failed}")
    else:
        log.info(f"✅ Auto-retry completado. Todos los reintentos exitosos.")


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
    """Transcribe todos los archivos en source/ (modo standalone)."""
    model = load_whisper_model()
    files = sorted([f for f in os.listdir(SOURCE) if f.lower().endswith(('.wav', '.mp3'))])
    total = len(files)
    log.info(f"📂 Archivos a transcribir: {total}")
    json_paths = []
    for i, f in enumerate(files, 1):
        json_path = process_whisper_file(f, model, progress, i, total)
        if json_path:
            json_paths.append(json_path)
    return json_paths


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
        r = requests.post(ACTIVE_WEBHOOK_URL, json=data, headers=headers, timeout=30)
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


def run_webhook(progress, files_override=None):
    """
    Envía JSONs al webhook.
    - Sin --file: procesa todos los JSONs en transcripts/ (lotes de 3 con cooldown)
    - Con --file: envía solo los archivos especificados, sin cooldown ni lotes
    """
    if files_override:
        total = len(files_override)
        log.info(f"🚀 Modo Webhook (archivos específicos) | {total} archivo(s)")
        for json_path in files_override:
            if not os.path.exists(json_path):
                log.error(f"❌ Archivo no encontrado: {json_path}")
                continue
            if not json_path.endswith('.json'):
                log.error(f"❌ No es un archivo JSON: {json_path}")
                continue
            # Usa el nombre del archivo como key en progress
            file_key = os.path.basename(json_path)
            update_progress(progress, file_key, "PENDING")
            send_webhook_local(json_path, file_key, progress)
        return

    # Comportamiento original: todos los JSONs en transcripts/
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
# PIPELINE INTERCALADO
# ==========================

def run_pipeline(ids, progress, include_video=False):
    """
    Pipeline intercalado por video:
      Para cada ID: descarga → transcribe → webhook → siguiente ID

    La transcripción actúa como cooldown natural entre descargas de YT.
    Al finalizar, ejecuta auto-retry para los IDs que fallaron en descarga.
    """
    check_cookies()
    total = len(ids)
    model = load_whisper_model()

    for i, video_id in enumerate(ids, 1):
        log.info("")
        log.info(f"{'─'*45}")
        log.info(f"  [{i}/{total}] Procesando → {video_id}")
        log.info(f"{'─'*45}")

        # ── 1. DESCARGA ───────────────────────────────────────
        if include_video:
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

        if not ok_audio:
            log.error(f"❌ [{i}/{total}] Audio FAILED → {video_id} | saltando transcripción y webhook")
            update_progress(progress, video_id, "FAILED", step="download-audio", error="yt-dlp error")
            continue  # sigue con el siguiente ID

        log.info(f"✅ [{i}/{total}] Audio OK → {video_id}")

        # ── 2. TRANSCRIPCIÓN ──────────────────────────────────
        audio_file = find_downloaded_audio(video_id)
        if not audio_file:
            log.error(f"❌ [{i}/{total}] No se encontró el WAV en source/ → {video_id}")
            update_progress(progress, video_id, "FAILED", step="transcribe", error="wav not found in source/")
            continue

        json_path = process_whisper_file(audio_file, model, progress, i, total)

        if not json_path:
            log.error(f"❌ [{i}/{total}] Transcripción falló → {video_id} | saltando webhook")
            continue

        # ── 3. WEBHOOK ────────────────────────────────────────
        send_webhook(json_path, progress)

    # ── AUTO-RETRY para descargas fallidas ────────────────────
    auto_retry_downloads(ids, progress, include_video=include_video)


# ==========================
# PIPELINE LOCAL (NAS)
# ==========================

LOCAL_VIDEO_EXTENSIONS = ('.mp4', '.mkv')


def load_local_files(args):
    """Recolecta archivos de video desde --file y/o --local-folder."""
    files = []

    if args.file:
        for f in args.file:
            if not os.path.exists(f):
                log.error(f"❌ Archivo no encontrado: {f}")
            elif not f.lower().endswith(LOCAL_VIDEO_EXTENSIONS):
                log.error(f"❌ Formato no soportado (solo mp4/mkv): {f}")
            else:
                files.append(os.path.abspath(f))

    if args.local_folder:
        folder = args.local_folder
        if not os.path.isdir(folder):
            log.error(f"❌ Carpeta no encontrada: {folder}")
        else:
            found = sorted([
                os.path.abspath(os.path.join(folder, f))
                for f in os.listdir(folder)
                if f.lower().endswith(LOCAL_VIDEO_EXTENSIONS)
            ])
            log.info(f"📂 Carpeta NAS: {len(found)} archivo(s) encontrados en {folder}")
            files.extend(found)

    if not files:
        log.error("❌ No se encontraron archivos válidos. Usá --file o --local-folder.")
        sys.exit(1)

    # Eliminar duplicados manteniendo orden
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return unique


def process_local_file(filepath, model, progress, index, total):
    """
    Transcribe un archivo de video local (mp4/mkv) directamente desde el NAS.
    - Usa el nombre del archivo como key en progress.json
    - Mueve el archivo original a processed/ dentro de su carpeta de origen
    """
    filename  = os.path.basename(filepath)
    base_name = os.path.splitext(filename)[0]
    json_path = os.path.join(TRANSCRIPTS, f"{base_name}.json")
    file_key  = filename  # key en progress.json es el nombre del archivo
    video_id  = get_clean_video_id(filename)  # extrae ID del nombre igual que en modo online

    log.info(f"🎧 [{index}/{total}] Transcribiendo (local) → {filename} (ID: {video_id})")
    update_progress(progress, file_key, "IN_PROGRESS", step="transcribe")

    try:
        segments, info = model.transcribe(filepath, language="es")

        result = {
            "videoId":  video_id,
            "filename": filename,
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

        print()
        log.info(f"   💾 Guardando transcripción → {json_path}")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        # Mover video original a processed/ dentro de su carpeta en el NAS
        nas_processed_dir = os.path.join(os.path.dirname(filepath), "processed")
        os.makedirs(nas_processed_dir, exist_ok=True)
        dest = os.path.join(nas_processed_dir, filename)
        shutil.move(filepath, dest)
        log.info(f"   📁 Video movido → {dest}")

        update_progress(progress, file_key, "DONE", step="transcribe")
        return json_path, file_key

    except Exception as e:
        log.error(f"❌ [{index}/{total}] Transcripción FAILED → {filename} | {e}")
        update_progress(progress, file_key, "FAILED", step="transcribe", error=e)
        return None, file_key


def send_webhook_local(json_path, file_key, progress):
    """Versión de send_webhook que usa file_key (nombre de archivo) en lugar de video_id."""
    filename = os.path.basename(json_path)
    log.info(f"🌐 Webhook → {filename}")
    update_progress(progress, file_key, "IN_PROGRESS", step="webhook")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        headers = {"X-Webhook-Secret": WEBHOOK_SECRET} if WEBHOOK_SECRET else {}
        r = requests.post(ACTIVE_WEBHOOK_URL, json=data, headers=headers, timeout=30)
        if r.status_code == 200:
            log.info(f"   ✅ Webhook OK → status {r.status_code}")
            shutil.move(json_path, os.path.join(TRANSCRIPTS_DONE, filename))
            log.info(f"   📁 Movido → transcripts/done/{filename}")
            update_progress(progress, file_key, "DONE", step="webhook")
        else:
            log.error(f"   ❌ Webhook respondió {r.status_code} → {filename}")
            update_progress(progress, file_key, "FAILED", step="webhook", error=f"status {r.status_code}")
    except Exception as e:
        log.error(f"   ❌ Webhook FAILED → {filename} | {e}")
        update_progress(progress, file_key, "FAILED", step="webhook", error=e)


def run_local_pipeline(files, progress):
    """
    Pipeline para archivos locales del NAS:
      Para cada archivo: transcribe → webhook → siguiente
    """
    total = len(files)
    model = load_whisper_model()

    for i, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        log.info("")
        log.info(f"{'─'*45}")
        log.info(f"  [{i}/{total}] Procesando (local) → {filename}")
        log.info(f"{'─'*45}")

        json_path, file_key = process_local_file(filepath, model, progress, i, total)

        if json_path:
            send_webhook_local(json_path, file_key, progress)
        else:
            log.error(f"❌ [{i}/{total}] Saltando webhook → {filename}")


# ==========================
# MAIN
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="ythelper: YouTube → Whisper → Webhook",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
modos:
  download-audio   Descarga solo audio WAV → source/
  download-video   Descarga solo video (máx calidad) → videos/
  download-all     Descarga audio + video en una pasada
  transcribe       Transcribe los archivos en source/
  webhook          Envía transcripts al webhook (lotes de 3)
  run-audio        Pipeline intercalado: audio → transcribe → webhook
  run-full         Pipeline intercalado: audio+video → transcribe → webhook
  local            Pipeline desde NAS: transcribe → webhook (sin descarga)

ejemplos:
  python ythelper.py --mode run-full  --ids abc123 def456
  python ythelper.py --mode run-audio --ids-file ids.txt
  python ythelper.py --mode run-full  --resume
  python ythelper.py --mode transcribe
  python ythelper.py --mode webhook
  python ythelper.py --mode local --file /mnt/nas/sermones/video.mp4
  python ythelper.py --mode local --local-folder /mnt/nas/sermones/
        """
    )
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
            "local",
        ],
        required=True,
        metavar="MODE",
        help="Modo de ejecución (ver lista abajo)"
    )
    parser.add_argument("--ids",      nargs="+", help="IDs de YouTube separados por espacio")
    parser.add_argument("--ids-file", help="Archivo .txt con IDs (uno por línea o varios por línea)")
    parser.add_argument(
        "--file",
        nargs="+",
        help="Uno o más archivos de video locales (mp4/mkv) para modo local"
    )
    parser.add_argument(
        "--local-folder",
        help="Carpeta del NAS — procesa todos los mp4/mkv que encuentre (modo local)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retomar desde progress.json, salteando los IDs ya DONE"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Usar WEBHOOK_URL_TEST del .env en lugar de WEBHOOK_URL (ambiente de pruebas)"
    )
    args = parser.parse_args()

    # Crear carpetas base
    for d in [SOURCE, VIDEOS, PROCESSED, TRANSCRIPTS, TRANSCRIPTS_DONE, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

    setup_logging()

    # Resolver URL de webhook según --test
    global ACTIVE_WEBHOOK_URL
    if args.test:
        if not WEBHOOK_URL_TEST:
            log.error("❌ --test activo pero WEBHOOK_URL_TEST no está definida en el .env")
            sys.exit(1)
        ACTIVE_WEBHOOK_URL = WEBHOOK_URL_TEST
        log.info(f"🧪 Modo TEST activo → {ACTIVE_WEBHOOK_URL}")
    else:
        if not WEBHOOK_URL:
            log.error("❌ WEBHOOK_URL no está definida en el .env")
            sys.exit(1)
        ACTIVE_WEBHOOK_URL = WEBHOOK_URL

    log.info(f"{'='*45}")
    log.info(f"  ythelper | modo: {args.mode} | test: {args.test} | resume: {args.resume}")
    log.info(f"{'='*45}")

    # Validación: local requiere --file o --local-folder
    if args.mode == "local" and not args.file and not args.local_folder:
        parser.error("--mode local requiere --file y/o --local-folder")

    progress = load_progress()

    needs_ids = args.mode in ("download-audio", "download-video", "download-all", "run-audio", "run-full")

    if needs_ids:
        if args.resume:
            ids, progress = load_ids_for_resume()
        else:
            ids = load_ids(args)
            for vid in ids:
                if vid not in progress:
                    update_progress(progress, vid, "PENDING")
        log.info(f"📋 IDs a procesar: {len(ids)}")

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
        if args.file:
            run_webhook(progress, files_override=args.file)
        else:
            run_webhook(progress)

    # ── RUN-AUDIO ─────────────────────────────────────────────
    elif args.mode == "run-audio":
        run_pipeline(ids, progress, include_video=False)

    # ── RUN-FULL ──────────────────────────────────────────────
    elif args.mode == "run-full":
        run_pipeline(ids, progress, include_video=True)

    # ── LOCAL ─────────────────────────────────────────────────
    elif args.mode == "local":
        files = load_local_files(args)
        log.info(f"📋 Archivos a procesar: {len(files)}")
        for f in files:
            if os.path.basename(f) not in progress:
                update_progress(progress, os.path.basename(f), "PENDING")
        run_local_pipeline(files, progress)

    print_summary(progress)
    log.info("✨ ¡Todo terminado!")


if __name__ == "__main__":
    main()

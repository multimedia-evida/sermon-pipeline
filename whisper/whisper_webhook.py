import os
import json
import shutil
import requests
import argparse
import time
import sys
from dotenv import load_dotenv

load_dotenv()

# Configuración de rutas
SOURCE = "source"
PROCESSED = "processed"
TRANSCRIPTS = "transcripts"
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
MODEL_SIZE = "small"

def send_webhook(json_path):
    filename = os.path.basename(json_path)
    print(f"🌐 Webhook → {filename}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        r = requests.post(WEBHOOK_URL, json=data, timeout=30)
        print(f"   ✅ Status: {r.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def get_clean_video_id(filename):
    """Extrae el ID de 11 caracteres después del primer guion bajo."""
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_', 1)
    if len(parts) >= 2:
        return parts[1][:11] # Los 11 chars del ID de YouTube
    return base_name

def process_whisper_file(filename, model):
    base_name = os.path.splitext(filename)[0]
    video_id = get_clean_video_id(filename)
    filepath = os.path.join(SOURCE, filename)
    json_path = os.path.join(TRANSCRIPTS, f"{base_name}.json")

    print(f"\n🎧 Whisper → {filename} (ID: {video_id})")

    # Transcripción con logs de progreso
    segments, info = model.transcribe(filepath, language="es")
    
    result = {
        "videoId": video_id,
        "language": info.language,
        "duration": info.duration,
        "segments": []
    }

    # Procesar segmentos y mostrar progreso en la terminal
    for segment in segments:
        # Log de progreso en la misma línea
        percent = (segment.end / info.duration) * 100
        sys.stdout.write(f"\r   ⏳ Progreso: {percent:.2f}% | {segment.end:.1f}/{info.duration:.1f} seg")
        sys.stdout.flush()

        result["segments"].append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    
    print(f"\n   💾 Guardando transcripción...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    shutil.move(filepath, os.path.join(PROCESSED, filename))
    return json_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["webhook", "whisper", "both"], required=True)
    args = parser.parse_args()

    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(TRANSCRIPTS, exist_ok=True)

    # 1. MODO WEBHOOK (Lotes de 3 con pausa)
    if args.mode == "webhook":
        files = sorted([f for f in os.listdir(TRANSCRIPTS) if f.endswith('.json')])
        print(f"🚀 Modo Webhook | {len(files)} archivos")
        
        for i in range(0, len(files), 3):
            batch = files[i:i+3]
            print(f"\n📦 Lote {i//3 + 1}: {batch}")
            for f in batch:
                send_webhook(os.path.join(TRANSCRIPTS, f))
            
            # Si no es el último lote, esperamos 2 minutos
            if i + 3 < len(files):
                print(f"⏳ Cooldown: esperando 2 minutos...")
                time.sleep(120)

    # 2. MODOS WHISPER O BOTH (Secuencial para evitar bloqueos)
    else:
        from faster_whisper import WhisperModel
        print(f"🚀 Cargando modelo {MODEL_SIZE}...")
        model = WhisperModel(MODEL_SIZE, compute_type="int8", cpu_threads=4)
        
        files = sorted([f for f in os.listdir(SOURCE) if f.lower().endswith(('.wav', '.mp3'))])
        print(f"📂 Archivos a procesar: {len(files)}")

        for f in files:
            json_path = process_whisper_file(f, model)
            
            if args.mode == "both":
                send_webhook(json_path)

    print("\n✨ ¡Todo terminado!")

if __name__ == "__main__":
    main()

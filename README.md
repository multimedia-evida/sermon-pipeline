# ythelper

Pipeline de automatización para descarga de audio/video desde YouTube, transcripción con Whisper y envío de resultados a un webhook.

```
YouTube → yt-dlp → Whisper (faster-whisper) → Webhook (n8n / cualquier endpoint)
```

---

## ¿Qué hace?

1. **Descarga** audio WAV y/o video en máxima calidad desde YouTube usando `yt-dlp`
2. **Transcribe** el audio con `faster-whisper` (modelo local, sin API externa)
3. **Envía** la transcripción como JSON a un webhook con autenticación por header
4. **Registra** el progreso de cada video en `logs/progress.json` y en un archivo de log por ejecución
5. **Reintenta automáticamente** las descargas fallidas (hasta 2 veces, con 5 minutos de pausa)

---

## Requisitos

- Python 3.9+
- [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) instalado y disponible en el PATH
- [`ffmpeg`](https://ffmpeg.org/) instalado (requerido por yt-dlp para conversión de audio)
- Cookies de YouTube exportadas (ver sección [Cookies](#cookies))

### Dependencias Python

```bash
pip install faster-whisper requests python-dotenv
```

---

## Instalación

```bash
git clone https://github.com/tu-usuario/sermon-pipeline.git
cd sermon-pipeline
pip install -r requirements.txt
cp .env.example .env
# Editar .env con tus valores
```

---

## Configuración

Crear un archivo `.env` en la raíz del proyecto:

```env
WEBHOOK_URL=https://tu-servidor/webhook/endpoint
WEBHOOK_SECRET=tu_secret_aqui
YT_COOKIES_PATH=/ruta/a/cookies.txt
```

| Variable | Descripción |
|---|---|
| `WEBHOOK_URL` | URL del endpoint que recibirá las transcripciones |
| `WEBHOOK_SECRET` | Valor enviado en el header `X-Webhook-Secret` para autenticación |
| `YT_COOKIES_PATH` | Ruta absoluta al archivo de cookies de YouTube (formato Netscape) |

---

## Cookies

YouTube requiere cookies para descargar ciertos videos (especialmente streams privados o con restricciones de edad). Para exportarlas:

1. Instalar la extensión [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) en Chrome
2. Ir a [youtube.com](https://youtube.com) con tu cuenta iniciada
3. Exportar las cookies en formato Netscape
4. Guardar el archivo y apuntar `YT_COOKIES_PATH` a esa ruta

---

## Modos de uso

```bash
python ythelper.py --mode MODE [--ids ID ...] [--ids-file FILE] [--resume]
```

### Modos disponibles

| Modo | Descripción |
|---|---|
| `download-audio` | Descarga solo audio WAV → `source/` |
| `download-video` | Descarga solo video en máxima calidad → `videos/` |
| `download-all` | Descarga audio + video en una sola pasada |
| `transcribe` | Transcribe los archivos `.wav`/`.mp3` que haya en `source/` |
| `webhook` | Envía los JSONs de `transcripts/` al webhook (en lotes de 3) |
| `run-audio` | **Pipeline completo:** descarga audio → transcribe → webhook (intercalado por video) |
| `run-full` | **Pipeline completo:** descarga audio + video → transcribe → webhook (intercalado por video) |

### Ejemplos

```bash
# Pipeline completo para un video
python ythelper.py --mode run-audio --ids dBp8VW3rAdQ

# Pipeline completo para varios videos
python ythelper.py --mode run-full --ids abc123 def456 xyz789

# Pipeline completo desde un archivo de IDs
python ythelper.py --mode run-full --ids-file ids.txt

# Solo descargar audio
python ythelper.py --mode download-audio --ids dBp8VW3rAdQ

# Solo transcribir los archivos que ya están en source/
python ythelper.py --mode transcribe

# Solo enviar webhooks de los JSONs pendientes en transcripts/
python ythelper.py --mode webhook

# Retomar una ejecución anterior (saltea los ya completados)
python ythelper.py --mode run-full --resume
```

### Archivo de IDs (`--ids-file`)

Archivo de texto plano, uno o varios IDs por línea. Las líneas que empiezan con `#` se ignoran.

```text
# Tanda del lunes
dBp8VW3rAdQ
k-HJKYrnQPQ

# Tanda del martes — varios en una línea
5YguWbAXBmc EKSOT7A7ghc 1kwue3QCOyo
```

---

## Estructura de carpetas

```
sermon-pipeline/
├── ythelper.py
├── .env
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
│
└── data/
    ├── input/
    │   ├── ids.txt                  # (opcional) archivo de IDs a procesar
    │   ├── source/                  # Audios WAV listos para transcribir
    │   └── videos/                  # Videos descargados (máxima calidad)
    │
    ├── output/
    │   ├── processed/               # Audios ya transcriptos (movidos desde source/)
    │   └── transcripts/             # JSONs de transcripciones pendientes de enviar
    │       └── done/                # JSONs enviados exitosamente al webhook (status 200)
    │
    └── logs/
        ├── 20260313_143200.log      # Log por ejecución con timestamps
        └── progress.json            # Estado de cada video ID
```

---

## Pipeline intercalado y anti-bloqueo de YouTube

Los modos `run-audio` y `run-full` procesan cada video de forma **intercalada**:

```
Video 1: descarga → transcribe → webhook
Video 2: descarga → transcribe → webhook   ← la transcripción del video 1 fue el cooldown natural
Video 3: descarga → transcribe → webhook
...
```

La transcripción (que tarda varios minutos por video) actúa como pausa natural entre descargas, evitando que YouTube detecte el patrón de ráfaga y bloquee las solicitudes.

### Auto-retry de descargas fallidas

Al finalizar el loop principal, el script detecta automáticamente los IDs que fallaron en la descarga y ejecuta hasta **2 rondas de reintento** con **5 minutos de pausa** entre cada intento individual. Si el reintento tiene éxito, completa también la transcripción y el webhook para ese video.

```
[fin del loop principal]
AUTO-RETRY ronda 1/2 | 3 ID(s)
  🔄 Reintentando → abc123
  ⏳ Esperando 5 min...
  🔄 Reintentando → def456
  ...
AUTO-RETRY ronda 2/2 | 1 ID(s)
  ...
```

---

## Tracking de progreso

Cada video pasa por los siguientes estados en `logs/progress.json`:

```
PENDING → IN_PROGRESS → DONE
                      ↘ FAILED (con step y error)
```

```json
{
    "dBp8VW3rAdQ": {
        "status": "DONE",
        "step": "webhook",
        "updated_at": "2026-03-13 14:32:01"
    },
    "k-HJKYrnQPQ": {
        "status": "FAILED",
        "step": "download-audio",
        "error": "yt-dlp error",
        "updated_at": "2026-03-13 14:35:22"
    }
}
```

El flag `--resume` lee este archivo y saltea los IDs con `status: DONE`, permitiendo retomar una ejecución interrumpida sin reprocesar lo que ya está completo.

---

## Formato del webhook

El script envía un `POST` a `WEBHOOK_URL` con el header `X-Webhook-Secret` y el siguiente body JSON:

```json
{
    "videoId": "dBp8VW3rAdQ",
    "language": "es",
    "duration": 2272.35,
    "segments": [
        {
            "start": 0.0,
            "end": 12.84,
            "text": "texto transcripto del segmento"
        }
    ]
}
```

Si el endpoint responde con `200`, el JSON se mueve automáticamente a `transcripts/done/`. Cualquier otro status code se registra como `FAILED` en el progress.

---

## Resumen al finalizar

Al terminar cada ejecución se imprime un resumen en pantalla y en el log:

```
=============================================
  RESUMEN FINAL
  Total   : 52
  ✅ OK    : 49
  ❌ Falló : 3
=============================================
  ❌ abc123 | step: download-video | error: yt-dlp error
  ❌ def456 | step: webhook        | error: status 500
  → Podés reintentar con --resume
=============================================
```

---

## Licencia

MIT

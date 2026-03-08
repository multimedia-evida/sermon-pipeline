# YouTube Audio Downloader

This script downloads audio from specific YouTube videos and converts it to **WAV format** using `yt-dlp`.

It is designed to batch-download sermon recordings or media content that will later be processed in an automated pipeline such as transcription with Whisper or additional audio processing.

---

## Purpose

The script automates the process of downloading multiple YouTube videos and extracting their audio in high-quality WAV format.

Typical use cases include:

* Downloading sermon recordings
* Preparing audio for transcription (Whisper)
* Creating a local archive of church media
* Feeding audio into automated processing pipelines

---

## Requirements

The following tools must be installed:

* **yt-dlp**
* **ffmpeg**
* **bash**

### Ubuntu / Debian

```bash
sudo apt install ffmpeg
pip install yt-dlp
```

### macOS

```bash
brew install ffmpeg
pip install yt-dlp
```

### Verify installation

```bash
yt-dlp --version
ffmpeg -version
```

---

## Configuration

Edit the script and define the list of **YouTube video IDs**.

Example:

```bash
IDS="oc_y5cwBHz8 AIy1HZC1eAg"
```

Each ID corresponds to a YouTube video.

Example URL:

```
https://www.youtube.com/watch?v=oc_y5cwBHz8
```

---

## Output Directory

Downloaded files will be saved in:

```
downloads/
```

The filename format is:

```
%(upload_date)s_%(id)s_%(title)s.%(ext)s
```

Example output file:

```
20240721_oc_y5cwBHz8_Sermon_Title.wav
```

---

## Running the Script

Make the script executable:

```bash
chmod +x download_audio.sh
```

Run the script:

```bash
./download_audio.sh
```

The script will:

1. Iterate through all video IDs
2. Download each video
3. Extract the audio
4. Convert it to WAV format
5. Save the file in the output directory

---

## Cookie Authentication (Optional)

The script supports **YouTube cookies** for videos that require authentication.

Example configuration in the script:

```bash
--cookies /path/to/cookies.txt
```

You can export cookies from your browser using extensions such as:

* Get cookies.txt
* EditThisCookie

Example path used in the script:

```
/home/lucho/repo/yt-download/www.youtube.com_cookies.txt
```

---

## Example Output

```
=====================================
Descargando ID: oc_y5cwBHz8
=====================================

✅ Descarga completada para oc_y5cwBHz8

🎉 Proceso finalizado.
```

---

## Integration with Processing Pipeline

This script is typically used as the **first step of the sermon media processing pipeline**:

```
YouTube
   ↓
Audio Download (this script)
   ↓
Whisper Transcription
   ↓
Content Processing
   ↓
Publishing / Archiving
```

---

## Notes

* The script downloads **audio only**
* Output format is **WAV (high quality)**
* Playlists are ignored
* Each video is processed individually
* The script logs success or failure for each download

---

## Possible Improvements

Future enhancements may include:

* Automatic extraction of video IDs from playlists
* Parallel downloads
* Integration with cloud storage (Google Drive, S3, etc.)
* Automatic triggering of transcription pipelines
* Logging system for large batch downloads

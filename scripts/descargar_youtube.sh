#!/bin/bash

# ==========================
# CONFIGURACIÓN
# ==========================

# 👉 PONÉ TUS IDs ACÁ (separados por espacio)
#IDS="5YguWbAXBmc EKSOT7A7ghc 1kwue3QCOyo qMf1yNY_qV0 1yB1_Na91NY"
IDS="k-HJKYrnQPQ"


# Carpeta destino
OUTPUT_DIR="downloads"

# ==========================
# NO TOCAR DEBAJO
# ==========================

mkdir -p "$OUTPUT_DIR"

for ID in $IDS; do
  echo "====================================="
  echo "Descargando ID: $ID"
  echo "====================================="

  yt-dlp --cookies /home/lucho/repo/yt-download/www.youtube.com_cookies.txt  "https://www.youtube.com/watch?v=$ID" \
    -f "bv*+ba/b" \
    --extract-audio \
    --audio-format wav \
    --audio-quality 0 \
    --no-playlist \
    -o "$OUTPUT_DIR/%(upload_date)s_%(id)s_%(title)s.%(ext)s"

  if [ $? -eq 0 ]; then
    echo "✅ Descarga completada para $ID"
  else
    echo "❌ Error al descargar $ID"
  fi

  echo ""
done

echo "🎉 Proceso finalizado."

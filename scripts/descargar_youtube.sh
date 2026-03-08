#!/bin/bash

# ==========================
# CONFIGURACIÓN
# ==========================

# 👉 PONÉ TUS IDs ACÁ (separados por espacio)
#IDS="AIy1HZC1eAg 2Plrti8K_50 AIy1HZC1eAg ZgV-u55pb6c cK5gaCARAFU OKMmZ7nv8OM CWADTGSreZA XD8omAYEn6U WBSadyezqeo CO1_VVOVyY0 ey8XTaboEOk l5ppMiZ5F0c JUszV6j3fr8 prqxGLIwkN8 AkGgIb0dU-o n4wReAaLfxs 6i1vYJrLMiA LDoCMk2V3_Y EaDjNuDNch8 PfcbV1WTyHs KbFDQeUfpog cxmDU_H-RsY yfueA4tIQDI udiER_JR6F4 itoyW_5xHK4 qFI6Pn0d648 G9l_Gmaggis dBp8VW3rAdQ QbsORsF2itE ObOXWHv8OhU kHo_qL1MkW0 z_r4cZkvdTY znNmHckKdgY 8KGjrEHhtyA Tw05L_0Jxgs uVtHXh3o2uk cFXbKaS3ZNs 9SdvF3ECKkU fqVvXX6LU8E lDfqqkl-spU Y7gUWCcCKls rhnR50KNDEM y1c4qEVEiHs 5YguWbAXBmc EKSOT7A7ghc 1kwue3QCOyo qMf1yNY_qV0 1yB1_Na91NY"
#IDS="E8kqKjSXI_E"
#IDS="AIy1HZC1eAg l5ppMiZ5F0c JUszV6j3fr8 prqxGLIwkN8 AkGgIb0dU-o n4wReAaLfxs 6i1vYJrLMiA LDoCMk2V3_Y EaDjNuDNch8 PfcbV1WTyHs KbFDQeUfpog cxmDU_H-RsY yfueA4tIQDI udiER_JR6F4 itoyW_5xHK4 qFI6Pn0d648 G9l_Gmaggis dBp8VW3rAdQ QbsORsF2itE ObOXWHv8OhU kHo_qL1MkW0 z_r4cZkvdTY znNmHckKdgY 8KGjrEHhtyA Tw05L_0Jxgs uVtHXh3o2uk cFXbKaS3ZNs 9SdvF3ECKkU fqVvXX6LU8E lDfqqkl-spU Y7gUWCcCKls rhnR50KNDEM y1c4qEVEiHs 5YguWbAXBmc EKSOT7A7ghc 1kwue3QCOyo qMf1yNY_qV0 1yB1_Na91NY"
#IDS="5YguWbAXBmc EKSOT7A7ghc 1kwue3QCOyo qMf1yNY_qV0 1yB1_Na91NY"
IDS="oc_y5cwBHz8"
#❌ Error al descargar AIy1HZC1eAg


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

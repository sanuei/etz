#!/usr/bin/env bash
# Mux the rendered picture with a language mix into the delivery files.
# usage: encode.sh [cn|en]
set -euo pipefail
cd "$(dirname "$0")/.."
lang=${1:-cn}
pic=build/segments/picture.mkv
[ "$lang" = cn ] || pic=build/segments_${lang}/picture.mkv
mix=build/audio/mix_${lang}.wav
out=output
mkdir -p "$out" build/pass build/deliver
tag="ETZ_LightBeyondBoundaries_${lang^^}"
common=(-pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tune grain -profile:v high -level 4.1)
# master: 2-pass ~9 Mbps H.264 (keeps the film grain), AAC 320k
ffmpeg -y -loglevel error -i "$pic" -c:v libx264 -preset slow -b:v 9M -maxrate 14M -bufsize 20M -pass 1 \
  -passlogfile build/pass/master "${common[@]}" -an -f mp4 /dev/null
ffmpeg -y -loglevel error -i "$pic" -i "$mix" -map 0:v -map 1:a -c:v libx264 -preset slow -b:v 9M -maxrate 14M \
  -bufsize 20M -pass 2 -passlogfile build/pass/master "${common[@]}" -c:a aac -b:a 320k -ar 48000 \
  -movflags +faststart -shortest "$out/${tag}_1080p.mp4"
# preview: 2-pass ~3.4 Mbps, stays under 30 MB for chat apps (kept out of git, in build/deliver)
ffmpeg -y -loglevel error -i "$pic" -c:v libx264 -preset slow -b:v 3400k -maxrate 5M -bufsize 7M -pass 1 \
  -passlogfile build/pass/preview "${common[@]}" -an -f mp4 /dev/null
ffmpeg -y -loglevel error -i "$pic" -i "$mix" -map 0:v -map 1:a -c:v libx264 -preset slow -b:v 3400k -maxrate 5M \
  -bufsize 7M -pass 2 -passlogfile build/pass/preview "${common[@]}" -c:a aac -b:a 160k -ar 48000 \
  -movflags +faststart -shortest "build/deliver/${tag}_preview.mp4"
ls -la "$out" build/deliver

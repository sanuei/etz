#!/usr/bin/env bash
# Mux the V2 picture with the score + narration mix. usage: encode2.sh [cn|en]
set -euo pipefail
cd "$(dirname "$0")/.."
lang=${1:-cn}
pic=build/v2_segments/picture.mkv
[ "$lang" = cn ] || pic=build/v2_segments_${lang}/picture.mkv
mix=build/audio/mix_${lang}.wav
tag="ETZ_LightBeyondBoundaries_V2_${lang^^}"
mkdir -p output build/pass build/deliver
common=(-pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tune grain -profile:v high -level 4.1)
ffmpeg -y -loglevel error -i "$pic" -c:v libx264 -preset slow -b:v 9M -maxrate 14M -bufsize 20M -pass 1 \
  -passlogfile build/pass/v2master "${common[@]}" -an -f mp4 /dev/null
ffmpeg -y -loglevel error -i "$pic" -i "$mix" -map 0:v -map 1:a -c:v libx264 -preset slow -b:v 9M -maxrate 14M \
  -bufsize 20M -pass 2 -passlogfile build/pass/v2master "${common[@]}" -c:a aac -b:a 320k -ar 48000 \
  -movflags +faststart -shortest "output/${tag}_1080p.mp4"
ffmpeg -y -loglevel error -i "$pic" -c:v libx264 -preset slow -b:v 3400k -maxrate 5M -bufsize 7M -pass 1 \
  -passlogfile build/pass/v2preview "${common[@]}" -an -f mp4 /dev/null
ffmpeg -y -loglevel error -i "$pic" -i "$mix" -map 0:v -map 1:a -c:v libx264 -preset slow -b:v 3400k -maxrate 5M \
  -bufsize 7M -pass 2 -passlogfile build/pass/v2preview "${common[@]}" -c:a aac -b:a 160k -ar 48000 \
  -movflags +faststart -shortest "build/deliver/${tag}_preview.mp4"
ls -la output build/deliver

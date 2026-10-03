#!/usr/bin/env bash
# Fonts used for titles/subtitles (SIL Open Font License), fetched from google/fonts.
set -euo pipefail
cd "$(dirname "$0")/../assets/fonts"
base=https://raw.githubusercontent.com/google/fonts/main/ofl
curl -fsSL -o 'Montserrat[wght].ttf' "$base/montserrat/Montserrat%5Bwght%5D.ttf"
curl -fsSL -o 'NotoSansSC[wght].ttf' "$base/notosanssc/NotoSansSC%5Bwght%5D.ttf"
curl -fsSL -o 'NotoSerifSC[wght].ttf' "$base/notoserifsc/NotoSerifSC%5Bwght%5D.ttf"
echo fonts ready

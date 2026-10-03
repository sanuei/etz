#!/usr/bin/env bash
# NASA Earth textures (public domain) used by the V2 planet renderer.
set -euo pipefail
cd "$(dirname "$0")/../assets/textures" 2>/dev/null || { mkdir -p "$(dirname "$0")/../assets/textures"; cd "$(dirname "$0")/../assets/textures"; }
curl -fsSL -o blackmarble_3km.jpg "https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144898/BlackMarble_2016_3km.jpg"
curl -fsSL -o bluemarble_5400.jpg "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg"
curl -fsSL -o clouds_2048.jpg "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57747/cloud_combined_2048.jpg"
curl -fsSL -o earth_specular_2048.jpg "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_specular_2048.jpg"
echo textures ready

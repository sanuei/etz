"""Render the whole film: parallel chunks -> lossless-ish segments -> final mp4.

usage: render.py [lang=cn] [--from SEC] [--to SEC]
"""
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
BUILD = os.path.join(ROOT, "build", "segments")  # per-language dir chosen in main()
FPS = 24
TOTAL = 60 * FPS
CHUNK = 48


def render_chunk(args):
    k0, k1, build = args
    import cv2
    cv2.setNumThreads(1)
    import film
    path = os.path.join(build, f"seg_{k0:05d}.mkv")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return k0, 0.0
    tmp = path + ".part.mkv"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "6", "-pix_fmt", "yuv444p",
           tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for k in range(k0, k1):
        f = film.render_frame(k / FPS)
        p.stdin.write((np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    os.replace(tmp, path)
    return k0, (time.time() - t0) / (k1 - k0)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    lang = args[0] if args else "cn"
    os.environ["ETZ_LANG"] = lang
    build = BUILD if lang == "cn" else BUILD + "_" + lang
    os.makedirs(build, exist_ok=True)
    chunks = [(k, min(TOTAL, k + CHUNK), build) for k in range(0, TOTAL, CHUNK)]
    t0 = time.time()
    with Pool(4) as pool:
        for k0, spf in pool.imap_unordered(render_chunk, chunks):
            print(f"chunk {k0 / FPS:5.1f}s done  {spf:4.2f}s/frame  elapsed {time.time() - t0:6.0f}s", flush=True)
    lst = os.path.join(build, "list.txt")
    with open(lst, "w") as f:
        for k0, _, _ in chunks:
            f.write(f"file 'seg_{k0:05d}.mkv'\n")
    master = os.path.join(build, "picture.mkv")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", master],
                   check=True)
    print("picture:", master)


if __name__ == "__main__":
    main()

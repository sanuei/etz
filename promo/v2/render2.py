"""Render V2: parallel chunks (Numba single-threaded per worker) -> segments -> picture.mkv
usage: render2.py [cn|en]"""
import os
import subprocess
import sys
import time
from multiprocessing import Pool

os.environ["NUMBA_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FPS = 24
TOTAL = 60 * FPS
CHUNK = 24


def render_chunk(args):
    k0, k1, build = args
    import cv2
    cv2.setNumThreads(1)
    import film2
    path = os.path.join(build, f"seg_{k0:05d}.mkv")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return k0, 0.0
    tmp = path + ".part.mkv"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "6", "-pix_fmt", "yuv444p", tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for k in range(k0, k1):
        f = film2.render_frame(k / FPS)
        p.stdin.write((np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    os.replace(tmp, path)
    return k0, (time.time() - t0) / (k1 - k0)


def main():
    lang = sys.argv[1] if len(sys.argv) > 1 else "cn"
    os.environ["ETZ_LANG"] = lang
    build = os.path.join(ROOT, "build", "v2_segments" + ("" if lang == "cn" else "_" + lang))
    os.makedirs(build, exist_ok=True)
    # heavy shots first so the pool stays balanced
    chunks = [(k, min(TOTAL, k + CHUNK), build) for k in range(0, TOTAL, CHUNK)]
    heavy = lambda c: -(1 if (c[0] / FPS < 10.5 or 31.5 <= c[0] / FPS < 34.5 or c[0] / FPS >= 47.0) else 0)
    order = sorted(chunks, key=heavy)
    t0 = time.time()
    with Pool(4) as pool:
        for k0, spf in pool.imap_unordered(render_chunk, order):
            print(f"chunk {k0 / FPS:5.1f}s done  {spf:5.2f}s/frame  elapsed {time.time() - t0:6.0f}s", flush=True)
    lst = os.path.join(build, "list.txt")
    with open(lst, "w") as f:
        for k0, _, _ in chunks:
            f.write(f"file 'seg_{k0:05d}.mkv'\n")
    out = os.path.join(build, "picture.mkv")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out],
                   check=True)
    print("picture:", out, flush=True)


if __name__ == "__main__":
    main()

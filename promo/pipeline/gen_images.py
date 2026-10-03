"""Generate the film's key frames with MiniMax image-01 (2048x864 ≈ 2.39:1)."""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from mm_client import image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "images", "raw")
EDL = json.load(open(os.path.join(HERE, "..", "script", "edl.json")))


def job(args):
    sid, prompt, n, tag = args
    paths = [os.path.join(OUT, f"{sid}_{tag}{k}.jpg") for k in range(n)]
    if all(os.path.exists(p) for p in paths):
        return sid, "cached"
    try:
        image(prompt, paths)
        return sid, "ok"
    except Exception as e:  # keep going; report at the end
        return sid, f"ERR {e}"


def main(ids, n, tag, extra=""):
    os.makedirs(OUT, exist_ok=True)
    jobs = [(sid, EDL["image_prompts"][sid] + extra, n, tag) for sid in ids]
    with ThreadPoolExecutor(3) as ex:
        for sid, status in ex.map(job, jobs):
            print(sid, status, flush=True)


if __name__ == "__main__":
    ids = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else list(EDL["image_prompts"])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    tag = sys.argv[3] if len(sys.argv) > 3 else "a"
    main(ids, n, tag)

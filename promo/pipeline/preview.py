"""Render still frames at given timestamps for review: preview.py out_dir t1 t2 ..."""
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np


def work(args):
    out_dir, T = args
    import film
    t0 = time.time()
    f = film.render_frame(T)
    p = os.path.join(out_dir, f"f_{T:06.2f}.jpg")
    cv2.imwrite(p, (np.clip(f, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return T, time.time() - t0


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    ts = [float(x) for x in sys.argv[2:]]
    with Pool(4) as pool:
        for T, dt in pool.imap(work, [(out, T) for T in ts]):
            print(f"{T:6.2f}s  {dt:5.2f}s/frame", flush=True)

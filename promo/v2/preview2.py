"""Render V2 stills: preview2.py out_dir t1 t2 ...  (ETZ_FAST=1 for quick looks)"""
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402


def work(args):
    out_dir, T = args
    cv2.setNumThreads(1)
    import film2
    t0 = time.time()
    f = film2.render_frame(T)
    cv2.imwrite(os.path.join(out_dir, f"f_{T:06.2f}.jpg"), (np.clip(f, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    return T, time.time() - t0


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    ts = [float(x) for x in sys.argv[2:]]
    with Pool(4) as pool:
        for T, dt in pool.imap_unordered(work, [(out, T) for T in ts]):
            print(f"{T:6.2f}s  {dt:6.2f}s", flush=True)

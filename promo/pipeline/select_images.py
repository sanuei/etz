"""Pick the chosen image-01 takes and clean small text artifacts in corners."""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "assets", "images", "raw")
SEL = os.path.join(HERE, "..", "assets", "images", "sel")

PICKS = {  # shot plate -> chosen take
    "S01": "S01_a1", "S02": "S02_a0", "S03": "S03_a1", "S04": "S04_b1", "S05": "S05_a1",
    "S06": "S06_c2", "S07": "S07_a1", "S08": "S08_a1", "S09": "S09_a1", "S10": "S10_b0",
    "S11": "S11_a1", "S12": "S12_a0", "S13": "S13_a0", "S14": "S14_a0", "S16": "S16_a1",
    "S17": "S17_a0", "S18": "S18_a1",
}
# (x0, y0, x1, y1) boxes holding stray generated "watermark" glyphs
CLEAN = {
    "S01": [(1890, 838, 2048, 864)],
    "S03": [(1975, 846, 2048, 864)],
    "S08": [(1940, 838, 2048, 864)],
    "S10": [(1900, 826, 2048, 864)],
}


def main():
    os.makedirs(SEL, exist_ok=True)
    for shot, take in PICKS.items():
        img = cv2.imread(os.path.join(RAW, take + ".jpg"))
        if shot in CLEAN:
            mask = np.zeros(img.shape[:2], np.uint8)
            for x0, y0, x1, y1 in CLEAN[shot]:
                mask[y0:y1, x0:x1] = 255
            img = cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.join(SEL, shot + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 96])
        print(shot, take, img.shape)


if __name__ == "__main__":
    main()

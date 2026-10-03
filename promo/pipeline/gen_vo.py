"""Generate the narration with MiniMax Speech-2.8-HD (one file per line)."""
import json
import os
import sys

from mm_client import tts

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "audio", "vo")

# Lines written by MiniMax-M3.1 (script/), lightly polished for the cut.
LINES_CN = [
    ("V1", "在宇宙诞生之前，<#0.45#>只有寂静。"),
    ("V2", "光，<#0.3#>找到了方向。"),
    ("V3", "价值，<#0.3#>开始流动。"),
    ("V4", "现实与远方，<#0.25#>此刻相连。"),
    ("V5", "随时随地，<#0.25#>交易无界。"),
    ("V6", "昼夜流转，<#0.25#>连接全球。"),
    ("V7", "安全，<#0.3#>是每一次跃迁的坐标。"),
    ("V8", "(iː)(tiː)(zɛd)。<#0.5#>开启无限未来金融。"),  # IPA so "ETZ" is read as letters
]
LINES_EN = [
    ("V1", "Before the universe was born, <#0.4#>there was only silence."),
    ("V2", "Light <#0.25#>found its direction."),
    ("V3", "Value <#0.25#>began to flow."),
    ("V4", "Reality and the far beyond, <#0.25#>connected now."),
    ("V5", "Trade anytime. <#0.25#>Anywhere."),
    ("V6", "Day and night, <#0.25#>across the globe."),
    ("V7", "Security <#0.25#>is the coordinate of every leap."),
    ("V8", "(iː)(tiː)(ziː). <#0.5#>Open unlimited future finance."),
]
SPEED_OVERRIDE = {"en": {"V1": 0.96}}  # keep EN V1 clear of the big-bang hit at 5.25 s
VOICE = {
    "cn": dict(voice_id="Chinese (Mandarin)_Male_Announcer", speed=0.9,
               voice_modify={"pitch": -15, "intensity": -20, "timbre": -25}),
    "en": dict(voice_id="English_Trustworthy_Man", speed=0.88,
               voice_modify={"pitch": -20, "intensity": -20, "timbre": -25}),
}


def main(lang, only=None):
    os.makedirs(os.path.join(OUT, lang), exist_ok=True)
    mp = os.path.join(OUT, lang, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    for vid, text in (LINES_CN if lang == "cn" else LINES_EN):
        if only and vid not in only:
            continue
        p = os.path.join(OUT, lang, f"{vid}.wav")
        kw = dict(VOICE[lang])
        kw["speed"] = SPEED_OVERRIDE.get(lang, {}).get(vid, kw["speed"])
        info = tts(text, p, **kw)
        meta[vid] = {"text": text, "ms": info.get("audio_length")}
        print(vid, meta[vid], flush=True)
    with open(mp, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # usage: gen_vo.py [cn|en ...] [--only V8,V2]
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = set(args[i + 1].split(","))
        args = args[:i] + args[i + 2:]
    for lang in args or ["cn"]:
        main(lang, only)

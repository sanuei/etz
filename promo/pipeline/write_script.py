"""Ask MiniMax (best available language model) to write the film's script."""
import json
import os
import sys

from mm_client import chat

MODEL = os.environ.get("MM_TEXT_MODEL", "MiniMax-M3.1-Flash-Preview")

SYSTEM = """你是一位顶级的电影预告片导演兼文案（参考《星际穿越》《沙丘》《2001太空漫游》的气质），
擅长为科技与金融品牌打造史诗级、电影质感的品牌宣传片。你的文字克制、有力、有诗意，绝不浮夸空洞。"""

BRIEF = """为加密数字资产交易平台 ETZ（官网 etz.com）创作一支「太空宇宙」主题的品牌宣传片，要震撼、要有电影大片的高级质感。

【品牌事实（只能使用这些，不要编造数据）】
- ETZ：数字资产交易平台。产品：现货交易（如 BTC/USDT）、RWA 现实世界资产、App 随时随地交易、7×24 全球服务、资产安全保障。
- 官方 slogan：Connect Digital Assets, Open Unlimited Future Finance；Trade Anytime, Anywhere。
- 品牌色：翡翠绿（emerald green）+ 黑色。Logo 是由几块倾斜的几何切片组成的立体 "E" 形标志。
- 合规：不得承诺收益、不得出现"稳赚""暴富""保本"等字样。

【成片规格】
- 时长 60 秒，2.39:1 宽银幕，预告片三幕式结构：冷开场（奇点/宇宙诞生）→ 连接（光与能量的流动=价值的流动，地球、网络、星系）→ 跃迁（极速、安全、全球）→ 高潮汇聚为 ETZ Logo → 片尾 slogan + etz.com。
- 旁白：普通话，深沉有磁性的男声，电影预告片语气。总共 7~9 句，每句短（4~16 个字），句与句之间留出音乐与画面的呼吸。旁白总朗读时长控制在 30 秒以内。
- 画面：全部为写实的太空场景（星云、星系、黑洞、地球夜景、轨道空间站、超空间跃迁、星座连线等），可以用"星座连成K线""光流汇聚成网络"等隐喻表达交易与连接，但要高级、克制。
- 画面制作方式：每个镜头由一张超高清电影剧照 + 摄影机运动（推/拉/摇/环绕/视差）+ 粒子光效实现，因此每个镜头描述要聚焦"一张能成立的史诗级画面 + 一个明确的运镜"。

【请输出 JSON（不要输出任何 JSON 以外的内容），结构如下】
{
  "title_cn": "...", "title_en": "...", "logline": "...",
  "music": "整体配乐与声音设计方向（一段话）",
  "vo": [ {"id": "V1", "cn": "旁白中文", "en": "英文翻译", "start": 秒, "emotion": "calm/surprised/..."} ],
  "supers": [ {"cn": "屏幕字幕/大标题中文", "en": "英文", "start": 秒, "end": 秒} ],
  "shots": [ {"id": "S01", "start": 秒, "end": 秒, "act": "冷开场/连接/跃迁/高潮/片尾",
              "visual": "画面内容（中文，具体到构图、光线、色彩）",
              "camera": "运镜", "fx": "粒子/光效/转场", "sound": "声音设计点",
              "image_prompt": "给图像模型的英文提示词：photorealistic IMAX film still ... (不要出现文字和logo)"} ]
}
要求镜头 14~18 个，节奏前慢后快，高潮处镜头 1~2 秒一切。"""


def main(out_dir):
    text, thinking = chat([{"role": "user", "content": BRIEF}], system=SYSTEM, model=MODEL, max_tokens=64000)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"script_raw_{MODEL}.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(out_dir, f"script_thinking_{MODEL}.txt"), "w") as f:
        f.write(thinking)
    s = text[text.find("{"): text.rfind("}") + 1]
    data = json.loads(s)
    with open(os.path.join(out_dir, f"script_{MODEL}.json"), "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../script")

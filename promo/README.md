# ETZ 品牌宣传片《光跃无界 / LIGHT BEYOND BOUNDARIES》

60 秒太空宇宙主题品牌片，2.39:1 宽银幕（1920×1080 带遮幅），24 fps，中文旁白 + 中英双语字幕。

| 文件 | 说明 |
| --- | --- |
| `output/ETZ_LightBeyondBoundaries_CN_1080p.mp4` | 主版本：中文旁白，H.264 约 9 Mbps，AAC 320k |
| `output/ETZ_LightBeyondBoundaries_EN_1080p.mp4` | 英文旁白版（同一画面，字幕按英文旁白计时） |
| `output/poster.jpg` | 片尾定版海报帧 |

`pipeline/encode.sh` 还会在 `build/deliver/` 下输出约 3.4 Mbps、小于 30 MB 的预览版，适合微信和社媒转发。该目录不入库。

响度按网络平台标准母带处理：−14.5 LUFS，真峰值 ≤ −1.2 dBTP。

## MiniMax 模型分工

| 环节 | 模型 | 产出 |
| --- | --- | --- |
| 剧本 / 分镜 / 文案 / 画面提示词 | **MiniMax-M3.1-Flash-Preview**（最新语言模型，`effort=max`） | `script/script_MiniMax-M3.1-Flash-Preview.json`：片名、8 句旁白、18 个镜头、配乐方向 |
| 关键帧画面 | **image-01**（2048×864，约 2.37:1） | 41 张候选，精选 17 张：`assets/images/sel/` |
| 旁白 | **Speech-2.8-HD** | 中文：`Chinese (Mandarin)_Male_Announcer` + 低沉音色调制；英文：`English_Trustworthy_Man` |
| 旁白质检 | **asr-1.0**（MiniMax 语音识别） | 逐句回听校验；"ETZ" 用 IPA 注音 `(iː)(tiː)(zɛd)` 修正读音 |

这把 Token Plan 订阅 Key 调不到的能力（已实测）：

- **MiniMax-H3 / H3-Max 视频**：`2068 当前 Token Plan 套餐等级不支持该模型`
- **Hailuo-2.3 / Hailuo-02 视频**：`2056 已达到 Token Plan 用量上限`（视频额度为 0）
- **music-3.0 音乐**：`2153 This Music API is no longer available to new users`

所以运动镜头和配乐由本仓库的代码完成：

- **镜头运动与特效**（`pipeline/fx.py`、`pipeline/film.py`）：在 image-01 画面上做亚像素运镜和运动模糊；黑洞吸积盘做开普勒差速旋转，星云做湍流流动；网络光路的光脉冲沿画面里已有的绿色光线自动寻路传播；另有 3D 星空超空间跃迁、变形宽银幕镜头光晕、体积光、粒子；手机屏幕里透视贴合了一套动态 K 线 App 界面；Logo 由 etz.com 官网 SVG 矢量还原，做碎片汇聚和扫光定版；最后是 ACES 色调映射、青绿调色、光晕溢散、色差、暗角和胶片颗粒。
- **配乐与声音设计**（`pipeline/score.py`）：80 BPM、D 小调，Logo 定版处转 D 大调。乐器有管风琴、合唱 Pad、玻璃质感琶音、低音弦乐 Ostinato、太鼓、Braam 重音、上升音、呼啸声和反向吸气。每个重音都对齐剪辑点。旁白处配乐自动闪避，最后做真峰值限幅母带。

如果升级到含 H3 的套餐（M Plan Explore / Build），可以把 `assets/images/sel/` 里的每张关键帧作为 H3 图生视频的首帧，生成真正的动态镜头替换进同一条时间线，分镜、旁白、字幕和配乐都不用改。

## 时间线（80 BPM，1 拍 = 0.75 s）

| 时间 | 镜头 | 旁白 | 字幕卡 |
| --- | --- | --- | --- |
| 0.00 | 黑洞奇点，从黑暗中浮现 | 在宇宙诞生之前，只有寂静。 | |
| 5.25 | 大爆炸白闪 → 新星诞生，后拉 | 光，找到了方向。 | |
| 10.50 | 能量光流汇聚，横移 | 价值，开始流动。 | |
| 15.00 | 地球夜面极光，城市光弧升空 | 现实与远方，此刻相连。 | RWA · 现实世界资产 |
| 20.25 | 轨道空间站，光脉冲沿航线汇入 | | |
| 24.00 | 星座连成 K 线（每根蜡烛对应一个音符） | 随时随地，交易无界。 | 现货交易 · BTC/USDT |
| 28.50 | 轨道节点网络 | | |
| 31.50 | 超大质量黑洞环绕 | | |
| 34.50 | 地平线日出，光脉冲环绕地球 | 昼夜流转，连接全球。 | 7×24 · 全球服务 |
| 37.50 | 舱内手持手机，App 行情界面 | | App 随时随地交易 |
| 39.75 | 飞船与能量护盾 | 安全，是每一次跃迁的坐标。 | 资产安全保障 |
| 42.00 | 超空间跃迁，白场 | | |
| 44.25 / 45.75 / 47.25 | 三连重击 | | 极速 / 连接 / 全球 |
| 47.25 | 日食背景，Logo 碎片汇聚；49.0 起停顿半拍 | | |
| 49.50 | Logo 锁定：闪光、冲击波、Braam | ETZ。开启无限未来金融。 | |
| 53.25 | 片尾定版 | | 连接数字资产，开启无限未来金融 · etz.com |

## 复现

```bash
pip install numpy opencv-python-headless pillow scipy scikit-image requests pyloudnorm
pipeline/fetch_fonts.sh                        # Montserrat / Noto Sans SC / Noto Serif SC (OFL)
export MINIMAX_API_KEY=...                     # Token Plan 订阅 Key（不要提交进仓库）

python pipeline/write_script.py script          # MiniMax-M3.1 写剧本（已生成，可跳过）
python pipeline/gen_vo.py cn en                 # Speech-2.8-HD 旁白（已生成）
python pipeline/gen_images.py all 2 a           # image-01 关键帧（已生成）
python pipeline/select_images.py                # 选片 + 清理角落瑕疵
python pipeline/score.py cn && python pipeline/score.py en   # 配乐 + 混音
python pipeline/render.py                       # 4 进程渲染 1440 帧
pipeline/encode.sh cn && pipeline/encode.sh en  # 封装交付文件
```

`pipeline/preview.py build/prev 5.2 26.5 49.7` 可以快速渲染单帧预览。

字体：Montserrat、Noto Sans SC、Noto Serif SC（SIL Open Font License）。Logo 几何数据取自 etz.com 官网的 SVG。

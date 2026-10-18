# AI-SonicMotion

用代码做音乐和配套动画：写好歌词与风格描述，用开源歌曲模型生成整首歌，分析节拍和每个唱词的时间，再逐帧渲染卡点 MV。

## 目录

| 路径 | 作用 |
|---|---|
| `works/<作品>/song.json` | 作品的单一事实源：风格描述、生成参数、音色参考、分段歌词（英文演唱 + 中文译文、大字歌词标记）、每段的画面场景与章节名 |
| `works/<作品>/timbre-reference.mp3` | 音色参考（用户选定的样例 B），合成时让音色向它靠拢 |
| `works/<作品>/composition-take7.json` | 选定的一版作曲（take 7 的模型规划与音频编码），可只换音色重新合成 |
| `music/setup_acestep.sh` | 拉取固定版本的 [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp)（ACE-Step 1.5 的 C++ 运行时）、编译、下载量化模型（约 7.7 GB） |
| `music/generate.py` | 每个随机种子生成一版完整的歌（take）；`--codes` 复用已有作曲只重新合成 |
| `music/analyze.py` | 节拍、底鼓、逐帧能量与 24 段频谱，以及唱词逐词时间（Whisper 识别后与歌词做全局对齐，重复的副歌按顺序对上，没识别到的词按前后插值） |
| `video/gfx.py` | 图形基元：可平铺噪声、透视相机、光点、泛光、体积光、镜头光晕、色调 |
| `video/scenes.py` | 各段场景（跑道、爬升入云、云上日落、盘旋、河与石、穿云下降、城市进近、落地），以及随节拍的光脉冲、雨、流星、星座连线、HUD 频谱环等 |
| `video/render.py` | 时间线与转场（闪光、漏光）、底鼓推镜与色散、字幕（唱到的词逐个点亮）、大字歌词逐词落下、章节标题、多进程渲染、ffmpeg 编码 |
| `outputs/` | 生成的音频、分析结果、静帧与视频（不入库） |

## 运行

macOS（Apple Silicon，自动使用 Metal）：

```bash
xcode-select --install && brew install cmake ffmpeg
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
bash music/setup_acestep.sh                                   # 首次：编译 + 下载模型

# 用音色参考重新合成选定的作曲（take 7），再分析、渲染
python3 music/generate.py works/ground-truth --seeds 7 --suffix _b --codes works/ground-truth/composition-take7.json
python3 music/analyze.py works/ground-truth outputs/ground-truth/takes/take_7_b.wav
python3 video/render.py works/ground-truth outputs/ground-truth/analysis/take_7_b.json

# 或者生成新的作曲（同样带音色参考），每个种子一版
python3 music/generate.py works/ground-truth --seeds 11 23 42
```

访问 Hugging Face 受限时，先 `export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`（模型与 Whisper 都走镜像，下载中断后重跑会续传）。字体下载失败时自动改用系统字体。

Linux 同样可用，走 CPU。上游默认只用一半核心，没有超线程的机器设 `ACE_THREADS=<核心数>`；4 核 CPU 生成一版约 3.5 分钟的歌要 30 分钟左右。

渲染预览：`--stills 12,48,95` 输出这几秒的 PNG；`--start 40 --end 70` 只渲染片段；`--width 960` 半分辨率。

## 决策记录

出歌方式、成片规格与个人信息边界见 [Agent Note：《Ground Truth · 落地》音乐与 MV 生成](.agents/notes/proposed/2026-10-18-ground-truth-music-video.md)。

## 许可与标注

- ACE-Step 1.5 模型为 MIT 许可，模型卡说明生成内容可商用。
- 字体 Cormorant Garamond、Noto Serif SC 为 OFL 许可，首次渲染时自动下载到 `.tools/fonts/`。
- 成片属于 AI 生成内容，发布时按平台要求标注；片尾署名已包含"AI 生成内容"。

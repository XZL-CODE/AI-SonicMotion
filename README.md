# AI-SonicMotion

用代码做音乐和配套动画：写好歌词与风格描述，用开源歌曲模型生成整首歌，分析节拍和每个唱词的时间，再逐帧渲染卡点 MV，最后为发布画封面。仓库是一级项目，每首歌是根目录下并列的一个二级项目，共用 `engine/` 里的生成、分析与渲染程序。

## 作品

| 目录 | 歌名 | 风格 | 当前版本 |
|---|---|---|---|
| [`ground-truth/`](ground-truth/README.md) | Ground Truth · 落地 | 深夜 R&B / neo-soul，英文演唱配中文字幕 | 作曲 take 7 + 样例 B 音色；MV 第二版；B 站与公众号封面 |

## 目录

| 路径 | 作用 |
|---|---|
| `<歌>/README.md` | 这首歌的歌名、简介、歌词、段落与画面、文件说明、重新生成命令 |
| `<歌>/song.json` | 这首歌的单一事实源：风格描述、生成参数、音色参考、分段歌词（英文演唱 + 中文译文、大字歌词标记）、每段的画面场景与章节名 |
| `<歌>/scenes.py` | 这首歌的画面：每个场景名对应一个逐帧绘制函数，可选 `FLASH_INTO`（哪些场景以一闪白光开场） |
| `<歌>/covers/` | 这首歌的发布封面：为它单独设计、手写的 SVG，以及导出的 JPG |
| `<歌>/outputs/` | 生成的音频、分析结果、静帧与视频（不入库） |
| `engine/music/setup_acestep.sh` | 拉取固定版本的 [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp)（ACE-Step 1.5 的 C++ 运行时）、编译、下载量化模型（约 7.7 GB） |
| `engine/music/generate.py` | 每个随机种子生成一版完整的歌（take）；`--codes` 复用已有作曲只重新合成 |
| `engine/music/analyze.py` | 节拍、底鼓、逐帧能量与 24 段频谱，以及唱词逐词时间（Whisper 识别后与歌词做全局对齐，重复的副歌按顺序对上，没识别到的词按前后插值） |
| `engine/video/gfx.py` | 图形基元：可平铺噪声、透视相机、光点、泛光、体积光、镜头光晕、色调 |
| `engine/video/render.py` | 载入这首歌的 `scenes.py`；时间线与转场（闪光、漏光）、底鼓推镜与色散、字幕（唱到的词逐个点亮）、大字歌词逐词落下、章节标题、多进程渲染、ffmpeg 编码 |
| `.tools/` | acestep.cpp、模型与字体（不入库） |

## 做一首新歌

每首新歌都走完下面几步。风格每首重新定，不沿用上一首：

1. 定风格：先分析这首歌的主题，写一段副歌；挑 3 个风格方向（2 个贴合主题，1 个刻意反差），每个方向用同一段副歌出一版约 30 秒的试听样曲，并配一句画面设想；听过、选定风格后再往下走。
2. 在根目录新建与 `ground-truth/` 并列的目录，写完整歌词和 `song.json`（风格、参数、分段歌词与译文、每段场景），以及这首歌自己的 `scenes.py`。
3. 生成、挑选、分析、渲染（命令见下节，把 `ground-truth` 换成新目录）。
4. 写这首歌的 `README.md`：歌名、简介、歌词，以及段落与画面、文件、重新生成命令。
5. 画封面：按这首歌的内容与画面单独设计，手写 SVG，导出 JPG，放在 `<歌>/covers/`，两种都要：
   - B 站：16:9，1920×1080。文字放在中间 4:3（x 240–1680），手机首页按这块裁切；底部两角留给播放量和时长角标。
   - 公众号：2.35:1，1410×600。所有信息放在中间 600×600（x 405–1005），转发卡片按 1:1 从中间裁；两侧只放画面。另导出这块正方形。
6. 在上面的作品表加一行。

## 运行

macOS（Apple Silicon，自动使用 Metal）：

```bash
xcode-select --install && brew install cmake ffmpeg
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
bash engine/music/setup_acestep.sh                             # 首次：编译 + 下载模型

# 用音色参考重新合成选定的作曲（take 7），再分析、渲染
python3 engine/music/generate.py ground-truth --seeds 7 --suffix _b --codes ground-truth/composition-take7.json
python3 engine/music/analyze.py ground-truth ground-truth/outputs/takes/take_7_b.wav
python3 engine/video/render.py ground-truth ground-truth/outputs/analysis/take_7_b.json

# 或者生成新的作曲（同样带音色参考），每个种子一版
python3 engine/music/generate.py ground-truth --seeds 11 23 42
```

访问 Hugging Face 受限时，先 `export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`（模型与 Whisper 都走镜像，下载中断后重跑会续传）。字体下载失败时自动改用系统字体。

Linux 同样可用，走 CPU。上游默认只用一半核心，没有超线程的机器设 `ACE_THREADS=<核心数>`；4 核 CPU 生成一版约 3.5 分钟的歌要 30 分钟左右。

渲染预览：`--stills 12,48,95` 输出这几秒的 PNG；`--start 40 --end 70` 只渲染片段；`--width 960` 半分辨率。

封面导出用无头 Chrome 截图再裁切（窗口要比画布高，否则底部会留白），macOS 上 `CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"`：

```bash
cd ground-truth/covers
"$CHROME" --headless=new --hide-scrollbars --window-size=1920,1300 --screenshot=/tmp/b.png "file://$PWD/bilibili.svg"
"$CHROME" --headless=new --hide-scrollbars --window-size=1410,800 --screenshot=/tmp/w.png "file://$PWD/wechat.svg"
ffmpeg -y -i /tmp/b.png -vf crop=1920:1080:0:0 -q:v 2 bilibili.jpg
ffmpeg -y -i /tmp/w.png -vf crop=1410:600:0:0 -q:v 2 wechat.jpg
ffmpeg -y -i /tmp/w.png -vf crop=600:600:405:0 -q:v 2 wechat-square.jpg
```

SVG 用 `.tools/fonts/` 里的字体（首次渲染时下载），没有时退回系统衬线字体。

## 决策记录

出歌方式、成片规格与个人信息边界见 [Agent Note：《Ground Truth · 落地》音乐与 MV 生成](.agents/notes/proposed/2026-10-18-ground-truth-music-video.md)。

## 许可与标注

- ACE-Step 1.5 模型为 MIT 许可，模型卡说明生成内容可商用。
- 字体 Cormorant Garamond、Noto Serif SC 为 OFL 许可，首次渲染时自动下载到 `.tools/fonts/`。
- 成片属于 AI 生成内容，发布时按平台要求标注；片尾署名已包含"AI 生成内容"。

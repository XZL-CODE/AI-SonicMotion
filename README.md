# AI-SonicMotion

用代码做音乐和配套动画：写好歌词与风格描述，用开源歌曲模型生成整首歌，分析节拍和每个唱词的时间，再逐帧渲染卡点 MV。

## 目录

| 路径 | 作用 |
|---|---|
| `works/<作品>/song.json` | 作品的单一事实源：风格描述、生成参数、分段歌词（英文演唱 + 中文译文）、每段对应的画面场景 |
| `music/setup_acestep.sh` | 拉取固定版本的 [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp)（ACE-Step 1.5 的 C++ 运行时）、编译、下载量化模型（约 7.7 GB） |
| `music/generate.py` | 每个随机种子生成一版完整的歌（take） |
| `music/analyze.py` | 节拍、底鼓、逐帧能量，以及唱词逐词时间（Whisper 识别后与歌词做全局对齐，重复的副歌按顺序对上，没识别到的词按前后插值） |
| `video/gfx.py` | 图形基元：可平铺噪声、透视相机、光点、泛光、色调 |
| `video/scenes.py` | 各段场景（跑道、爬升入云、云上日落、盘旋、河与石、穿云下降、城市进近、落地） |
| `video/render.py` | 时间线与转场、字幕（唱到的词逐个点亮）、多进程渲染、ffmpeg 编码 |
| `outputs/` | 生成的音频、分析结果、静帧与视频（不入库） |

## 运行

macOS（Apple Silicon，自动使用 Metal）：

```bash
brew install cmake ffmpeg
python3 -m pip install -r requirements.txt
bash music/setup_acestep.sh                                   # 首次：编译 + 下载模型
python3 music/generate.py works/ground-truth --seeds 7 11 23 42
python3 music/analyze.py works/ground-truth outputs/ground-truth/takes/take_11.wav
python3 video/render.py works/ground-truth outputs/ground-truth/analysis/take_11.json
```

Linux 同样可用，走 CPU。上游默认只用一半核心，没有超线程的机器设 `ACE_THREADS=<核心数>`；4 核 CPU 生成一版约 3.5 分钟的歌要 30 分钟左右。

渲染预览：`--stills 12,48,95` 输出这几秒的 PNG；`--start 40 --end 70` 只渲染片段；`--width 960` 半分辨率。

## 许可与标注

- ACE-Step 1.5 模型为 MIT 许可，模型卡说明生成内容可商用。
- 字体 Cormorant Garamond、Noto Serif SC 为 OFL 许可，首次渲染时自动下载到 `.tools/fonts/`。
- 成片属于 AI 生成内容，发布时按平台要求标注；片尾署名已包含"AI 生成内容"。

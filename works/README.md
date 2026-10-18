# 作品

每首歌一个目录 `works/<名字>/`：`README.md` 写歌名、简介与歌词，`song.json` 存风格描述、生成参数、分段歌词与画面场景，另放该作品用到的音色参考、选定作曲等输入。生成的音频和视频在 `outputs/<名字>/`，不入库。

| 目录 | 歌名 | 风格 | 时长 | 当前版本 |
|---|---|---|---|---|
| [ground-truth](ground-truth/README.md) | Ground Truth · 落地 | 深夜 R&B / neo-soul，英文演唱 | 3:34 | take 7 + 样例 B 音色，MV 第二版 |

`song.json` 里每段的 `scene` 对应 `video/scenes.py` 里的场景函数；新作品可以复用现有场景，也可以在那里加新场景。

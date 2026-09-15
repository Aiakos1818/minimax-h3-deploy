# 分镜方法与词库

## 1. 镜头表模板

| # | tag | 时长 | 生成方式 | 景别·机位·镜头 | 画面动作 | 光线色调 | 声音 | 转场 | 锚图 | seed |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | pingfan_balcony | 5s | 独立 | 大远景缓推中景 / 手持 | 他背身浇花 | 黄昏侧逆光·冷青灰+钠灯橙 | 水声/风/油炸 | 硬切 | balcony_best | 31001001 |
| S2 | pingfan_agespot | 5s | 独立 | 微距特写 / 极慢推 | 指腹抚摸手背老人斑 | 暖阳斜射皮肤 | 安静阳台/远处车流 | 硬切 | agespot_best | 31002002 |
| M1 | pingfan_ride | 3×5s | **链** | 长焦跟拍 / 稳定器 | 骑车穿城→车筐栀子花→手部特写 | 金色逆光·琥珀 | 链条/市声/蝉 | 硬切 | ride_best | 31004004 |

`tag` 进 `film_spec.json`；`生成方式` 决定用独立镜头还是链（见 §3）。

**命名规范**：`tag = <film>_<shot-slug>`，slug 用 2–3 个词描述**这一镜的内容**（地点/主体/动作，如 `balcony`/`agespot`/`ride`），**不要 `s1`/`m1` 这类序号**；一条链只用一个 tag；`anchor` key 复用同一 slug；镜头顺序只体现在 `jobs[]` 与 `assemble.clips[]`。

## 2. 拆 beat 的判据

1. **时空变化**（阳台 → 街上）→ 必然换镜（独立）。
2. **动作完成**（放下花洒 → 摸手背）→ 可换镜，也可一镜内完成（同一链内用 beat 推进）。
3. **情绪转折**（发现老年斑 → 回忆）→ 换镜 + 换 look；闪回另开一条链。
4. 一镜只放**一个动作或一个画面**；超过就是两镜。
5. 长短节奏：连续 3–4 个 5s 镜头后安排一个"停留镜头"（特写/空镜）呼吸。

## 3. 独立镜头 vs 链

- **独立镜头**（`--segments 1` + `--first-image`）：不同时空/景别/表情，硬切拼接。控制力强，重跑便宜。
- **链**（`--segments N --beat ... --merge`）：**同一动作连续推进**的长镜头（骑行、下车→走过来→上车）。
  链内每段画面强制连续，所以：
  - 基底 `--prompt` 只写共有时空/人物/光（同 run 不可改）；
  - 动作推进写 `--beat "<5k>s:<该段动作>"`（`seg = int(t // dur)`）；
  - 链首段才有 `--first-image` 锚图；链内每段会消耗一次新的 text encode（几秒到几十秒）。
- 想"一镜到底 + 换景"：不要用一条链硬扛，改交叉溶解接两条链。

## 4. LOOK 块库（每部片选 1–2 套，全片原样复用）

```
# 现实·黄昏冷调（手持）
Photorealistic live-action cinema, 35mm film, available dusk light, cool desaturated teal-grey
palette with warm sodium-lamp practical highlights, T2.0 shallow depth of field, gentle handheld
breathing, fine film grain, halation on highlights, restrained naturalistic grading,
1990s Chinese small-city realism, one continuous shot

# 记忆·金色暖调（稳定器）
Photorealistic live-action cinema, 35mm Kodak Vision3 250D, golden-hour backlight, warm amber and
honey highlights, bloom and halation, slightly blown sky, gentle grain, T2.0 shallow depth of
field, smooth Steadicam movement, nostalgic but real, one continuous shot

# 雨夜霓虹 / 黑白纪实 / 雪天冷灰（按需仿写：胶片 + 光比 + 调色 + 景深 + 机位 + 颗粒）
```

## 5. CHARACTER 块模板（逐字复用）

```
<name>, a <age>-year-old <ethnicity> <gender>, <脸/皱纹/发型>, <服装>,
<气质/情绪>, <年代与阶层感>, <in/out of a specific 年代 look>
```
例：`Li Guoqing, a 50-year-old Chinese man, sun-weathered face, deep crow's feet, short thinning black hair greying at the temples, plain grey-blue short-sleeve shirt, calm restrained melancholy, lived-in 1990s Chinese working-class look`

## 6. AUDIO 写法（H3 是音视频联合生成）

`AUDIO: <同期声 1>, <同期声 2>, no music.`／有台词时写 `no dialogue`。常用声景：水声、金属浇花壶、
炒锅油炸、窗纱扑动、蝉、自行车链条/车铃、厂区人流、远处车流、雨。旁白与配乐**不要**交给模型，后期配。

## 7. NEGATIVE 固定尾巴

```
No text, no subtitles, no captions, no logos, no watermarks, no animation or cartoon or game
rendering, no overly-CG look, no distorted hands, no extra fingers, no morphing or duplicated
faces, keep live-action texture and natural restrained motion.
```

## 8. 镜头语言词库（提示词直接用）

| 维度 | 词 |
|---|---|
| 景别 | extreme wide shot / wide / medium / medium close-up / close-up / extreme macro close-up / insert |
| 机位·运动 | static, pushes in very slowly, drifts back half a metre, handheld breathing, Steadicam glide, long-lens tracking shot, low angle beside the wheel, over-the-shoulder, crane |
| 光 | available dusk light, golden-hour backlight, warm sodium-lamp practical highlights, raking sunlight, soft window light, blown sky |
| 质感 | 35mm film, Kodak Vision3 250D, T2.0 shallow depth of field, fine/gentle film grain, halation, restrained naturalistic grading |
| 表演 | minimal motion, micro-expression, a small tired smile grows, his eyes lose focus, knuckles relaxed |

## 9. 时长与节奏

- 默认 `--dur 5`（净新 5s）；4–5s 最稳最快；7–9s 明显变慢且更容易漂。
- 独立镜头 5s；链按动作需要 2–4 段（10–20s）。
- 结尾：末帧定格 0.5–0.6s 再淡出（`assemble.hold` + `fade_out`），比直接切断更像片子。

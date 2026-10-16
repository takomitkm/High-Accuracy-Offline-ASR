# CrispASR-FireRed —— FireRedASR2-AED 批转链（crispasr，可上 N 卡）+ qwen3 兜底

驱动是 `Flid-Cg-firered-Cg-qwen.py`（曾用名按时间从早到晚：`xhs-asr.py` →
`crisper-xhs-qwen-asr.py` → `Wlid-Og-fired-Cg-qwen.py`（10-10~10-14）→ 现名，
全指的是同一个文件）。名字按仓库根 README 的记号法拆读：`Flid` = FireRedLID 判语种、
`Cg-fired` = 判到范围内就走 crispasr 的 FireRedASR2-AED（N 卡）、`Cg-qwen` = 判到范围外回退
crispasr 的 qwen3（N 卡）；主力一直是 FireRed 那套。

**这条链有两段实现，由驱动顶部一个常量挑（10-14 按她的指令把默认翻回 crispasr）：**

| `PRIMARY_LEG` | 主力那一路（"小红书 ASR"部分）由谁跑 | 语种怎么定 |
|---|---|---|
| `"crispasr"`（**出厂默认**） | CrispASR 的 `firered-asr` backend（AED 的 gguf q4_k 量化件，解码器恒在 CPU、编码器可上 CUDA） | **帧级路由**（10-16 起，`LID_ROUTER="frame"`）：驱动进程里常驻一份 FireRedLID ONNX，FireRedVAD 导段 → 整文件滑窗判语种 → 范围外满 3 段的文件**根本不喂 AED**。见第 7 节。设 `ASR_LID_ROUTER=sieve` 就退回 10-14 那版：**批内筛子**，一趟 `--lid-backend firered --lid-model firered-lid-f16.gguf`，从 stderr 逐文件读 `crispasr: LID -> language = 'xx' (firered, p=1.000)`，只看每条音频的头 5 秒 |
| `"onnx"`（`ASR_PRIMARY_LEG=onnx`） | 本包 `onnx/` 里的 **FireRedASR2S ONNX 全链路**：FireRedVAD → FireRedASR2-AED（纯 onnxruntime）→ FireRedPunc，三个会话在驱动进程里常驻一次 | **闸门**（独立两步，不用批内筛子）：crispasr 的默认 silero VAD 导语音段 → 拼前 15 s 语音 → `whisper-tiny -dl` 判语种。第 7 节那条帧级路由**不作用在这条腿上**（`ort` 原样不动） |

两档共用同一份兜底：判到 AED 语种范围外 → 这批文件立刻用 `qwen3` 重跑（同一批内完成）。
换档不改任何其它参数，onnx 那一条腿的代码一行没删。

形状没变的部分：AED 出**无标点**的字、标点由 `fireredpunc` 加（`"crispasr"` 档是 gguf 那份、
`"onnx"` 档是 ONNX 图的），VAD 用 `firered-vad`（`"crispasr"` 档是它的 gguf，`"onnx"` 档用它的
ONNX 图）。四件事各有各的文件，所以这一份是三个 CrispASR 方案里组件最多的一套
（crispasr 侧自动下载合计 **6,426,805,865 B**，10-14 加了筛子那件、10-16 把它从 q4_k
569,688,704 B 换成 f16 1,814,794,560 B；
`"onnx"` 档还要再加 `onnx/` 那三件权重，见第 1 节）。
10-16 加的帧级路由（第 7 节）还要 `onnx/models/FireRedLID-onnx` 那两张图（`lid_encoder.onnx`
2,451,530 + `lid_encoder.onnx.data` 2,893,425,664 + `lid_decoder_step.onnx` 549,729 +
`lid_decoder_step.onnx.data` 655,949,824 = **3,552,376,747 B**，逐件字节与 sha256 在 `cpu` 分支
那份 README 的导出表里，10-06 在校机上算的）外加同目录那对 `dict.txt` / `cmvn.ark`。这些都是
**建造产物**，`fetch_assets.py` 下不到它们；再要一个能跑的 ffmpeg。缺任何一样都在起批前当场
写在日志里并把路由降级成筛子，不会悄悄少判一片文件。

---

## 0. 整个逻辑（一段话）

驱动扫音频根目录、用 `<输出目录>\.lock` 抢单实例（两个驱动各载一份权重会抢同一块显存，故意互斥），每批先用 FireRedVAD（小红书自家，三档 VAD 里唯一不会把唱歌素材整条吞掉的那个）切段，然后跑**帧级语种路由** —— 驱动自己常驻一份 FireRedLID ONNX 会话（`onnx/optional-lid/lid_onnx.py`，纯 onnxruntime、providers 写死 CPU），把子段打包成 ≤60 s 的块各自编码（Conformer 全自注意力，代价 ∝ ΣTᵢ²，分块比整条更省），长 VAD 段先按 6 s 切成互不重叠的子段、每子段用 3 s 窗 / 1.5 s 跳投票，统计"判到 AED 语种范围外（中 / 英 / 粤语＋约 20 种汉语方言）"的子段数，**满 3 段独立证据才把整个文件扣下**（因为 crispasr 自己的前置判别器结构上只能看前 15 s、firered 适配器再截到 5 s，且 confidence 写死 `p=1.0`，没有置信度闸门可用）；没被扣下的文件一趟 `crispasr.exe` 批量调用（多个 `-f`，模型与 VAD 只加载一次）交给 FireRedASR2-AED（q4_k gguf，`-l auto`、`--lid-backend off`、`--chunk-seconds 20`），出**无标点**文本再经 FireRedPunc（q8_0）加书面标点，批内每 30 秒轮询已写出的 `.txt`、写完一个就落盘并清理源文件（中途被杀不再白跑）；被扣下的文件攒成第二批换 qwen3（2.51 GB q8_0）重跑，那一趟照旧 `-l auto` ＋前置判别筛子 —— 这个 build 的 qwen3 没有 `CAP_LANGUAGE_DETECT`，而驱动**故意不把自己判到的码传下去**（那等于把筛子的错误固化），且 qwen3 有能力位自己出标点，所以不再叠 punc；VAD 判全文件无语音的文件一律记进 `no_speech.txt`，`--strict-pipeline` 保证 VAD / 标点这类辅助环节加载失败时直接非零退出，而不是"warning 一声然后无标点静默跑完整棵树"；停止只在批次边界生效（Ctrl+C 一次＝当前批跑完后退出、两次＝立刻终止子进程，与建 `STOP` 文件或 `--stop` 等价），而子进程通过 Windows Job Object 绑在 python 进程上，**关窗口 / 强杀 python 会连坐杀掉 crispasr.exe**，所以只能用 `--stop`；降级路径三条 —— `ASR_LID_ROUTER=sieve` 回到 10-14 那版批内筛子（读 stderr 的 `crispasr: LID ->` 行对号挑文件，对不上号整批不采信）、`=off` 则 AED 一条道跑到黑、`ASR_PRIMARY_LEG=onnx` 则主链换成三个常驻 ORT 会话（FireRedVAD → AED → FireRedPunc）并启用 silero ＋ whisper-tiny `-dl` 15 秒闸门（帧级路由只在 crispasr 腿生效），另外路由那组权重（两张图连 `.onnx.data` 合计 3,552,376,747 B，是**建造产物**、不在任何下载清单里）或 ffmpeg 缺失时 `frame_router_setup()` 会整条降级成 sieve 并在日志里写清缺了什么。

每一段的依据、实测数字与已知洞在下面各节：复原 §1、跑 §2、组件与出处等级 §3、关键 CONFIG 值 §4、
已知洞与边界 §5、`onnx` 那条腿 §6、帧级路由 §7。

---

## 1. 复原

```bat
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python fetch_assets.py          :: crispasr 侧：下载 + 逐件 sha256 验收
.venv\Scripts\python onnx\fetch_assets.py --models onnx\models --graph mixed   :: ONNX 侧（PRIMARY_LEG="onnx" 才要）
```

`fetch_assets.py` 是通用的，读同目录 `assets.json`（URL、字节数、sha256、出处等级、许可证
都在里面）。落点默认在本目录下两个文件夹：

| 根 | 默认 | env | 装什么 |
|---|---|---|---|
| `bin` | `./crispasr` | `CRISPASR_BIN_DIR` | `crispasr.exe` + 7 个 dll + `crispasr-quantize.exe` + 文档 3 件 + CUDA 运行库 3 件（+ 4 个 MSVC 运行库 dll，需人工，见第 3 节） |
| `model` | `./model` | `CRISPASR_MODEL_DIR` | AED q4_k / FireRedPunc q8_0 / FireRedVAD / **firered-lid-f16（批内筛子）** / ggml-tiny（只有 `onnx` 档的闸门用）/ silero-v6.2.0（闸门）/ qwen3 q8_0 共 7 份权重 |

**ONNX 那三件权重不在 `assets.json` 里**，由 `onnx/fetch_assets.py` 自己那份清单管
（`python onnx/fetch_assets.py --list` 打印全表）。分两份是因为两侧的建造方式不同：
crispasr 侧全部是"下载即终件"，ONNX 侧有两件（`punc.f32.onnx`、`encoder.f32.onnx`）
是本地 `dequant_*.py` 从公开原料造出来的，逐字节可复现（10-06 从 int8/q8w 原料重跑，
五件产物与产线文件同 sha256）。`--graph` 决定造到哪一步，必须与驱动里的 `ONNX_GRAPH`
一致：默认 `mixed` = f32 编码器 + int8 解码器 = 建 `punc_f32` + `aed_encoder_f32`。
落盘约 5.5 GB，其中 `encoder.f32.onnx` 只有 934 KB、权重在旁边的 `.onnx.data` 里。
这一条不进库（`onnx/.gitignore` 的 `models/*`）。**10-14 起出厂默认腿是 crispasr，所以这一
步不再是"克隆后必须跑"** —— 只有要切 `ASR_PRIMARY_LEG=onnx` 时才跑它（切过去时驱动会在
起批前逐件检查，缺了就停，不会跑一半才发现）。

**默认腿对 model 根的要求**：`fetch_assets.py` 跑完（退出码 0）就齐了。缺 `firered-lid-f16.gguf`
时驱动在起批前 `logger.critical` + 退出码 2 停下（`-l auto` 挂了前置判别器就必须有它的本地文件，
留空或指不到文件会让 crispasr 每文件联网下载、离线机器每次卡满超时）；`ggml-tiny.bin` 和
`ggml-silero-v6.2.0.bin` 这两件在默认腿下【不参与拼命令】，缺了不影响 crispasr 那一路。
`--check` 是离线核对，退出码 3 = 有组件没就位。

想落到别处：设清单里写的同名环境变量（`CRISPASR_BIN_DIR` / `CRISPASR_MODEL_DIR`），下载与驱动读同一个值；
只想改下载落点就用 `--set-root model=<别的目录>`（相对按包目录算、绝对照用，驱动那边要用同一个值）。
核对现状不联网用 `--check`，看全部 URL 与哈希用 `--list`，只补某一件用 `--only NAME`。

上面那两个 env 就是驱动 CONFIG 里 `CRISPASR_BIN_DIR` / `MODEL_DIR` 读的两个；数据面另有
`ASR_TEXT_DIR` / `ASR_AUDIO_DIR`；`"onnx"` 档另有下表四个；帧级路由（第 7 节）另有两行。
都不设时全部落在包目录下，所以**复原完不改一行就能跑**：

| env | 默认（相对包目录） | 是什么 |
|---|---|---|
| `CRISPASR_BIN_DIR` | `./crispasr` | 上面 `bin` 那个根 |
| `CRISPASR_MODEL_DIR` | `./model` | 上面 `model` 那个根 |
| `ASR_TEXT_DIR` | `./txt` | 转写文本与日志的输出目录（`rules.txt`、`tmplist.txt`、`no_speech.txt`、`log\` 都在这） |
| `ASR_AUDIO_DIR` | `./audio` | 音频数据面根目录，驱动用它的 `p\` 当输入队列、`f\` 当失败隔离区、`t\` 当 ASCII 暂存区 |
| `ASR_PRIMARY_LEG` | `crispasr` | 主力那一路由谁跑：`crispasr` / `onnx`（见第 6 节）。10-14 起出厂默认是 `crispasr` |
| `FIREDASR_ONNX_MODELS_DIR` | `./onnx/models` | ONNX 三件权重的根目录 |
| `FIREDASR_ONNX_PROVIDER` | `cuda` | ONNX 的执行提供者：`cpu` / `cuda`。**只有这两个值**，见第 6 节那条"只用 N 卡"。没有 CUDA EP 的机器要显式给 `cpu`，否则起批前当场停 |
| `FIREDASR_GPU_PY` | 空 = 不换 | 装了 CUDA EP 的那个解释器的绝对路径。非空时驱动在起批前先看**当前解释器**有没有 `CUDAExecutionProvider`，没有就用它把自己原地重启一次（只一次，靠 `FIREDASR_GPU_PY_SWITCHED` 记账防循环）。出厂留空：仓库里不写机器专属路径，见 §6.3 |
| `FIREDASR_ONNX_GRAPH` | `mixed` | AED 图形状，必须与建权重时 `--graph` 给的那个一致 |
| `ASR_LID_ROUTER` | `frame` | 默认腿（`crispasr`）怎么挑语种：**`frame` = 整文件帧级路由（第 7 节）** / `sieve` = 10-14 那版批内筛子（只看头 5 秒）/ `off` = 不挑，AED 一条道跑到黑 |
| `ASR_FFMPEG` | 空 = 用 PATH 里的 `ffmpeg` | 帧级路由解 `m4a` / `mp4` / `aac` 用的可执行文件。libsndfile 不认这三种容器，而小红书下载下来的恰好主要是这一类；没有能跑的 ffmpeg，第 7 节那条路在起批前整体降级成 `sieve` |

七份 crispasr 侧权重的文件名分别写在 `CRISPASR_MODEL_AED` / `CRISPASR_MODEL_QWEN3` /
`CRISPASR_PUNC_MODEL` / `CRISPASR_LID_MODEL`（批内筛子 = firered-lid-f16）/
`GATE_WHISPER_MODEL` / `CRISPASR_VAD_MODEL` / `GATE_VAD_MODEL`
（见第 4 节），拼的都是 `MODEL_DIR` 下那几个名字，与 `fetch_assets.py` 的落点一一对得上。
值给相对路径就按包目录解析，给绝对路径就照用。`ASR_ROOT` / `ASRSOURCE` 必须在同一块盘上
（`t\` 用硬链接，跨盘会失败）。

## 2. 跑

```bat
.venv\Scripts\python Flid-Cg-firered-Cg-qwen.py --start   :: 分离后台启动
.venv\Scripts\python Flid-Cg-firered-Cg-qwen.py --stop     :: 写 STOP 标志，下一批边界干净退出
.venv\Scripts\python Flid-Cg-firered-Cg-qwen.py            :: 前台跑（Ctrl+C 一次=本批跑完退，两次=立刻杀子进程）
```

**关掉启动它的那个控制台 = 连坐杀 crispasr 子进程**（Job Object 是故意绑上去的，
用来防孤儿）。要停就用 `--stop`，别关窗。

**仓库这份是出厂形态**：default 全部相对包目录、env 优先，正文里没有一条机器专属路径。
笔记本桌面上那份（和校机分发包里那份）= 同一套代码，只把 default 换成那台机器的绝对路径、
注释里的 `<输出目录>` / `<音频根>` 记号换成实际目录。所以逻辑改动只改仓库这份，再往机器副本
重刷 default；反过来在副本里改逻辑，仓库后续的修正会静默丢掉（10-12 就对过一次性：桌面那四份
是 10-10 01:04 复制的，缺了之后的 20 s 分块、`GATE_LID_SEC` 定案、KV cache 说法更正和本节的
`GPU_PY`，已按上面这个口径重生成）。

I/O 契约（与 `CrispASR-Qwen/`、`cpu` 分支那两份逐条相同）：

- 递归扫 `ASRSOURCE\p` 下的音频（`.mp3 .m4a .mp4 .wav .oga .ogg .opus .flac .aac`），
  按**一级子目录分组**，组内按 mtime 升序；根上的散文件归 `p` 组。
- 输出 `ASR_ROOT\<组名>.txt`，**追加**写：`title:<相对 p 的路径>` + 正文 + 空行。
- 成功 send2trash；空转写即删且不落记录；失败按相对路径隔离进 `f\`（保目录结构）。
- `BATCH_SIZE = 28`；批内每 30 秒轮询已写出的 `.txt`，落盘一个清一个。
- 默认腿先跑**帧级路由**再跑 AED（第 7 节）：判到 AED 范围外满 3 个子段的文件**根本不喂 AED**，
  直接攒下来用 qwen3 兜底；放行的那趟 crispasr 不挂前置判别器（`--lid-backend off`）。
  用 `ASR_LID_ROUTER=sieve` 退回旧形态：语种筛子判到范围外的文件照样喂 AED、但输出**不落盘**，
  攒下来立刻用 qwen3 兜底重跑（同一批内完成）。
- VAD 判"整段无语音"的文件 rc 仍是 0、不落 `.txt`，转写树上什么都不留，
  所以在这里记账：`ASR_ROOT\no_speech.txt`（`NO_SPEECH_LOG`），格式是 `时间 \t 原因 \t 相对路径`。
- `.lock` 单实例锁，**与同一 `ASR_ROOT` 下的其它 crispasr 驱动互斥**（同一份数据面
  不能两个引擎同时吃）。
- `rules.txt`（同目录有 `rules.example.txt`）逐条容错，单条非法正则只跳过该行；
  `tmplist.txt` 是幻觉复读的黑名单。
- 收尾推送可选：`set NTFY_TOPIC_URL=https://…` 才发，不设就跳过（端点不入库）；
  Ctrl+N 在开/关之间切换，不必重启。

## 3. 组件清单与出处等级

`assets.json` 是机器可读的那份（`python fetch_assets.py --list` 打印全表）。摘要：

| 件 | 来源 | 等级 | 说明 |
|---|---|---|---|
| `crispasr.exe` + `crispasr.dll` + `crispasr-quantize.exe` + `ggml.dll` + `ggml-base.dll` + `ggml-cpu.dll` + `ggml-cuda.dll` + `whisper.dll` + LICENSE + `README-CUDA.txt` + THIRD_PARTY_NOTICES（zip 的 11 个成员） | GitHub release `v0.8.37/crispasr-windows-x86_64-cuda-non-cuda.zip`（146,153,022 B / `74d04c30…`） | **A** | zip 的 sha256 是 release 给的 digest；11 个成员各自的 sha256 从 zip 里算，且与作者生产机那份逐字节相同 |
| `cublas64_12.dll` / `cublasLt64_12.dll` / `cudart64_12.dll` | 同一个 release 页的独立资源（113,712,640 / 692,441,600 / 573,952 B） | **A** | 哈希取自 release 附带的 `crispasr-windows-x86_64-cuda-runtime-sha256.txt`（251 B / `92e42561…`） |
| `msvcp140/vcomp140/vcruntime140/vcruntime140_1.dll` | 微软 `vc_redist.x64.exe`（aka.ms 链接） | **C** | 装了 redist 就不用管；不能装就拷这 4 份到 exe 同目录。`--check` 只给 WARN |
| `firered-asr2-aed-q4_k.gguf` 962,807,328 B | `huggingface.co/cstr/firered-asr2-aed-GGUF` | **A** | sha256 = HF 的 LFS oid；也就是 CrispASR `-m auto` 注册表 `firered-asr` 那一行。上游 FireRedASR2 是 Apache-2.0 |
| `fireredpunc-q8_0.gguf` 108,664,800 B | `huggingface.co/cstr/fireredpunc-GGUF` | **A** | 书面标点后处理，**只给 AED 用**。必须给本地路径：留空时 crispasr 取 `auto` 并联网下 FireRedPunc（`examples/cli/crispasr_run.cpp:3940`），拉不动的机器每批白等一次超时。上游 BSD-2-Clause |
| `firered-vad.gguf` 2,357,952 B | `huggingface.co/cstr/firered-vad-GGUF` | **A** | **转写侧**的 VAD（`_base_cmd` 里 `-vm` 那一行，qwen3 兜底与容器退回都走它）。**转写不用 crispasr 默认的 silero v6.2.0**，理由见第 5 节；silero v6 在 `"onnx"` 档里只出现在闸门（见下面那行）。上游 BSD-2-Clause |
| `firered-lid-f16.gguf` 1,814,794,560 B | `huggingface.co/cstr/firered-lid-GGUF` | **A** | **批内语种筛子**（`--lid-backend firered --lid-model` 这件）。10-16 起出厂档从 q4_k 换成 **f16**，三条依据：①这就是 crispasr 自己的默认口径 —— 注册表 `lid-firered` 那一行（`src/crispasr_model_registry.cpp:1467-1474`）与 `--lid-backend firered` 不给 `--lid-model` 时自动下的那件（`examples/cli/crispasr_lid_cli.cpp:58-59`）都指 f16，而官方转换器 `models/convert-firered-lid-to-gguf.py` 只有 `--input/--output`、权重一律按 `float16` 落盘（`:146`、`:168`）⇒ f16 是**转换产物本身**，其余几档是 cstr 侧的量化件；②量化在这条路上**拿不到计算收益** —— 上游 `docs/ggml-optimisation-playbook.md:539-545` 实测，AVX2 的 EPYC 7763 上 q8_0 整个模型比 f32 还慢 9%（1.09×），要 0.72× 得靠 repack，而 repack 在 crispasr 里只接到 whisper（`src/crispasr.cpp:1925-1937`）与 hFT（`src/hft_transformer.cpp:391`）两处，FireRedLID 这条 backend 走不到，`repack.cpp:4528` 在 x86 上连 q8_0 分支都没有 ⇒ 换小档省下的只是磁盘和"每文件重装"的搬运体积；③判据没有置信度闸门可用（`p=` 恒 1.0，`src/firered_lid.cpp:90` 写死），精度只能靠权重保真。②③ 里"省不省时间、判得准不准"**都没量过**（标 C）。sha256 = HF 的 LFS oid（10-14 直读 `api/models/cstr/firered-lid-GGUF/tree/main`），10-16 又在生产机的模型目录里下载后逐字节核过、两头相同 ⇒ A 口径原先缺的那一半补齐了（10-14 时它是【无从对照】）。同仓另有 q4_k/q8_0/q2_k 三档，换档只改 path/bytes/sha256 三行（五档哈希都记在 `assets.json` 那条 note 里）；**8 GB 显存的机器按最便宜够用取 q4_k**（569,688,704 B）—— 筛子默认 `use_gpu=true`，f16 那 1.81 GB 会压在显存上。上游主 README 有两处可对照的口径：`:634` 把 q4_k 那档记作「Q4_K (544 MB), 120 languages including Chinese dialects. **Slower** but covers more languages」（那句 Slower 与 ② 同向 —— 上游也没把量化件当快的），`:516` 把 `--lid-backend firered` 的自动下载件记作 1.7 GB（= f16，与 ① 同向）。两处行号是 10-16 直读上游当日的行号；10-14 记的 `:611` / `:493` 已随上游漂移，**作废**。**四条代价**（窗口只有头 5 秒、置信度恒 1.0、失败不退回 whisper、每文件重装且 `use_gpu=true`）见第 5.2 节，本语料召回**未测** |
| `ggml-tiny.bin` 77,691,713 B | `huggingface.co/ggerganov/whisper.cpp` | **A** | 现在**只服务 `onnx` 档的语种闸门**（`-dl` 那一步，`GATE_WHISPER_MODEL`）。默认腿 crispasr 下它不参与拼命令、`onnx_leg_setup` 之外不检查它。切到 onnx 腿前必须把它下齐（要本地文件，否则 crispasr 联网下、离线机器卡满超时）。为什么闸门不能改用上面那件 firered-lid：`-dl` 的"判完就退出"和 `auto-detected language:` 那行都是 whisper 引擎特有的（`src/crispasr.cpp:7953-7970`） |
| `ggml-silero-v6.2.0.bin` 885,098 B | `huggingface.co/ggml-org/whisper-vad` | **A** | **闸门**第 1 步的 VAD，也就是 crispasr `--vad` 的默认那个（`examples/cli/crispasr_vad_cli.cpp:19` 的 URL 就是这条）。只用于"判语种前导个段"，不参与转写。sha256 两头对上：HF 的 LFS oid 与纯 CPU 测试机上实跑那份相同（10-09 `sha256sum` 直读） |
| `qwen3-asr-1.7b-q8_0.gguf` 2,506,723,200 B | `huggingface.co/cstr/qwen3-asr-1.7b-GGUF` | **A** | 兜底引擎权重。显存不够换同仓 q4_k（1,490,915,200 B / `ec197cef…`），换时 path 与 sha256 一起改 |
| `onnx/models/**`（AED 图 / FireRedVAD / FireRedPunc，约 5.5 GB） | `onnx/fetch_assets.py` 自己那份清单：sherpa-onnx 的 GitHub release + 两个 HuggingFace 仓，另有两件本地 `dequant_*` 建造 | **A**（下载件）/**A**（建造件按 sha256 逐字节复现） | `"onnx"` 档才需要。与 crispasr 侧那份清单**没有重叠**，所以不存在两处对齐问题；`--list` 打全表、`--check` 不联网核对现状 |

等级口径：**A** = 发布方给的哈希与作者生产机那份逐字节相同，照链接下就是同一个文件；
**B** = 哈希只取自生产机那份、没有独立凭据可对照；**C** = 没有公开匿名下载件，需人工。

### 3.1 上传前对"必须组件是否完善"做了什么

这台机器和校机都跑不了 CUDA（校机无独显、无 `nvidia-smi`），所以这一轮不是端到端实测，
而是三条静态/哈希级对账，全部通过：

1. **引用闭包**：驱动里 `os.path.join(CRISPASR_BIN_DIR/…, MODEL_DIR, "…")` 拼出来的运行时
   引用逐条抽出来（bin 侧 1 件 `crispasr.exe`、model 侧 5 件权重，与 155/158/160/162/167
   行一一对上），核对清单能否全部落地 —— 没有缺口。驱动自己创建的
   （`log\`、`.lock`、`STOP`、`t\`、`f\`、`no_speech.txt`）和要人手写的
   `rules.txt`/`tmplist.txt`/`<组名>.txt` 单独归类，不算组件。
2. **清单 ↔ 生产机目录**：`bin` 根 14 件、`model` 根 5 件的**字节数逐件相符**；
   清单能落 18 件里多出的 4 件正是 MSVC dll（生产机装了系统级 redist，exe 同目录没有它们，
   符合预期）。生产机 `crispasr-model` 目录另有 9 件，比这份清单多的 4 件是另外两条链用的
   （silero 两份、q4_k 那份、canary 对齐器），本链不引用。
3. **真算 sha256 的那几件**：拿 `fetch_assets.py --check --only …` 直接对生产机目录算过
   `firered-vad.gguf`（2.25 MB）、`fireredpunc-q8_0.gguf`（103.63 MB）、`ggml-tiny.bin`
   （74.09 MB）—— 都报"sha256 对上"。AED 962 MB 与 qwen3 q8_0 2.5 GB 本轮没有重算哈希，
   依据是发布方凭据（HF 的 LFS oid）+ 字节数相符，也就是上表的 A 级口径本身。

**这一节是 10-07 那轮的账，逐条照原始记录写着。** 10-14 清单加了第 7 件 model 侧权重、
10-16 又把它从 q4_k 换成 f16（`firered-lid-f16.gguf`，1,814,794,560 B，出处等级 A = HF 的 LFS oid，
见上表），上面这三条对账【没有为它重跑】：件数从 5 变 7，"引用闭包"和"清单 ↔ 生产机目录"两条的
数字都过期了。这一件 10-16 已经下到生产机的模型目录、sha256 与 LFS oid 逐字节相同（补上了 A 级
口径里原先缺的那一半）；但生产批次在跑的仍是 `cpu` 分支那条腿的驱动，**这件目前不被生产批次消费**。

**没有做过的**：这一份 CUDA 驱动在 GPU 上的端到端跑通。同一支 `crispasr.exe` 的 CPU 路径
已在纯 CPU 机器上跑通（`cpu` 分支 `Qwen3/` 第 3.1 节，含 rc=127→rc=0 那个 MSVC 门槛）。

## 4. 关键 CONFIG 值（为什么是这个值）

| 常量 | 值 | 为什么 |
|---|---|---|
| `ENGINES["aed"]` | backend `firered-asr` + `CRISPASR_MODEL_AED` + `CRISPASR_PUNC_MODEL` | 主引擎。AED 的中文信息保留度比 qwen3 稳（没有 LLM 解码器，结构上不可能把一段话概括成一句） |
| `ENGINES["qwen3"]` | backend `qwen3` + `CRISPASR_MODEL_QWEN3`，`punc` 留空 | 兜底档，只接筛子判到 AED 范围外的文件；qwen3 自己出标点，所以 `punc` 必须是空串 |
| `CRISPASR_GPU_BACKEND` | `"cuda"` | 改 `"cpu"` 就退回 CPU 构建那套（程序目录要一起换成 cpu zip） |
| `CRISPASR_THREADS` | `6` | CPU 侧解码线程。实测口径：**纯 CPU 时取 0.75 × 逻辑线程**；GPU 档这个值影响小 |
| `CRISPASR_LANGUAGE` | `"auto"` | AED 内置语种判别只覆盖 中文(+约20种汉语方言)/英语/粤语，范围外**不是差一点，是拿汉字编造**，而且 `-l` 改不了它 |
| `CRISPASR_LID_BACKEND` | `"firered"` | 前置筛子（10-14 从 `"whisper"` 换过来，配合 `CRISPASR_LID_MODEL` 那件 firered-lid-f16）。**只在默认腿 `crispasr` 下拼进转写命令**，每文件判一次；`onnx` 档不用它，那条腿的语种由下面 `GATE_*` 那三步独立判。四条性质与换回 whisper 的方法见第 5.2 节 |
| `BATCH_SIZE` | `28` | 一次调用喂多个 `-f`，模型/VAD 只加载一次 |
| `LID_ROUTER` | `"frame"` | 10-16 加的那层：换不换引擎由**整文件**的帧级判别决定，不是由前几秒决定。`frame` = 每条音频完整过一遍 FireRedLID ONNX（`onnx/optional-lid/lid_onnx.py`，CPU），按 VAD 段计票；`sieve` = 10-14 那种批内前置筛子（每文件只看前 5 s）；`off` = 两层都不挂，整批直接喂 AED。之所以非 `frame` 不可：`src/crispasr_lid.cpp:284` 把外部判别器的输入截到原始音频前 15 s，`src/firered_lid.cpp:78` 在里面**再截到 5 s**，而 crispasr 的 CLI **没有任何窗口旋钮**（只有 `--lid-backend` / `--lid-model` / `--lid-on-transcript`，`examples/cli/cli.cpp:496-500`）—— 也就是说"整个文件都在 FireRed 支持/不支持之列"这句话，crispasr 自己给不出来。判据与四条降级见 §7 |
| `FRAME_LID_MIN_BAD_SEGS` / `FRAME_LID_UNIT_SEC` / `FRAME_LID_MIN_UNIT_SEC` | 3 / 6.0 / 1.0 | 她定的"占比要到起码三段才切换 Qwen，以防误判"。**计数单位是子段不是 VAD 段**：一条日语独白在 VAD 下常常只有一两段（10-09 校机实测 80.2 s 素材的段表只有一条 0.03–79.55 s），不切子段的话"≥3 段"永远达不成；按 6 s 切成互不重叠的子段后，换引擎要三段独立证据 ≈ 18 s 范围外。短于 1 s 的尾巴凑不满一个 3 s 窗、只白占计数，丢掉。为什么门限是"段"而不是置信度：`src/firered_lid.cpp:90` 把 confidence **写死 1.0**，结构上没有置信度闸门可用 |
| `FRAME_LID_WIN_SEC` / `FRAME_LID_HOP_SEC` / `FRAME_LID_MODE` | 3.0 / 1.5 / `"slice"` | 滑窗逐窗解码，与 `onnx` 腿那条 `run_lid` 同一套参数、同一份代码（驱动直接调 `chain.run_lid`，不复制实现）。`slice` 是"一次编码、逐窗解码" |
| `FRAME_LID_SLICE_SEC` | 60.0 | 一次编码最多覆盖多少秒。编码器是 Conformer 全自注意力，分数矩阵随时长**平方**涨：十分钟素材 T=15000 帧（25 帧/s），单层 4 头就要 4×15000²×4 B ≈ 3.6 GB，夜间无人值守会被它撑死。把相邻子段打包成 ≤60 s 的块、每块各自编码，总代价 ∝ ΣTᵢ² 比整条**更省**；代价只有块边界上的窗少看几十秒上下文，而 3 s 窗判语种用不着长上下文。`0` = 不切，整条一次编码 |
| `FFMPEG_EXE` | PATH 里找到的 `ffmpeg`，找不到为空 | 路由要拿到 **PCM**，而 libsndfile 只认 wav/flac/ogg/oga/opus/mp3；`EXTENSIONS` 里的小红书主流封装 m4a / mp4 / aac 恰好全在外面。crispasr 内部那套 ffmpeg 是**静态链进二进制的库**，python 侧够不着，只能另找可执行文件。只在 libsndfile 失手时接管。**启动前探测（`_ffmpeg_probe()`）不通过 ⇒ 整条路由降级成 `sieve`**，因为"这台机器有没有能跑的 ffmpeg"是环境属性不是文件属性，而留着等于给这三类容器开了一个连筛子都没有的洞。要指定就 `ASR_FFMPEG=<路径>` |
| `PRIMARY_LEG` | `"crispasr"` | 主力那一路由谁跑（第 6 节）。10-14 按她的指令从 `"onnx"` 翻回来；要切 ONNX 腿设 `ASR_PRIMARY_LEG=onnx`，那条腿的代码与权重原样留着 |
| `ONNX_PROVIDER` | `"cuda"` | ONNX 的执行提供者，**只认 `cpu` / `cuda`**。写别的（DirectML、OpenVINO）驱动直接退出 —— 那两个能挑中 AMD 核显。出厂值 10-09 从 `cpu` 改成 `cuda`（同一批三件实测 cuda RTF 0.427 对纯 CPU 0.566，§6.6）。**没有 CUDA EP 的机器现在会当场停在起批前那道 `enc_ep` 断言**，要在那种机器上跑得显式给 `FIREDASR_ONNX_PROVIDER=cpu` |
| `GPU_PY` | `""`（不换） | 会话在本进程常驻（不外包子进程），所以"编码器在不在卡上"取决于**现在跑这份脚本的那个 python 里的 onnxruntime**。一台机器上常见"装 CUDA EP 的 ORT 在某个 venv 里，而双击 / 计划任务用的是另一个解释器"，这一步就是给那种情形留的：起批前换一次解释器，换完仍不起 CUDA 照旧停在 `enc_ep` 断言上。判据是纯函数 `should_switch_interpreter()`（只在无参数、`--start`、`--bg`、`/start` 这四条路上问 EP，`--stop` / `--help` / `crispasr` 主腿 / `provider=cpu` 一行 ORT 都不 import），逐条见 §6.3 |
| `ONNX_GRAPH` | `"mixed"` | f32 编码器 + int8 解码器。10-06 产线实测最快的那档，也是 `cuda` 下唯一有意义的选择（int8 图里那些整型算子 CUDA EP 跑不了） |
| `ONNX_DEC_ON_GPU` | `False` | 编码器上卡、解码器留 CPU。10-09 在 N 卡上实测：解码器也上卡**更慢**（整批 418.9 s → 537.2 s，+28%），显存倒是放得下（峰值 4,788 MiB / 8,188 MiB）。机制与 crispasr 那边同源结论见 §5.1，数值见 §6.6 |
| `ONNX_ASR_THREADS` / `ONNX_VAD_THREADS` / `ONNX_PUNC_THREADS` | 逻辑核一半 / ≤8 / 4 | 与 `cpu` 分支那个分发包同口径（10-06 实测并回去的） |
| `GATE_VAD_MODEL` | `model/ggml-silero-v6.2.0.bin` | 闸门第 1 步 = crispasr `--vad` 的**默认**那个模型（本地路径，不让它联网下） |
| `GATE_LID_SEC` / `GATE_MIN_SPEECH_SEC` | 15.0 / 1.0 | 判别用 15 s 语音。**这个 15 不是上游的限制**：闸门读的那行（whisper 自己的检测）实测吃**前 30 s**，而 crispasr 的**外部判别器**截 15 s（`src/crispasr_lid.cpp:284`，`kLidMaxSamples = 16000*15`）—— 那是批内筛子走的那条路，10-14 之后批内筛子换成了 FireRedLID，它在外层 15 s 之内**再截到 5 s**（`src/firered_lid.cpp:78`），见 §5.2。先前本节和驱动注释都写成"whisper 那条路径本来就只截 15 s"，**那是错的，在此公开更正**；窗口实测与后果见 §6.6 末条。**10-10 她已定：保持 15，不提到 30**（§6.7） |
| `GATE_THREADS` / `GATE_TIMEOUT_SEC` | 2 / 300 | 闸门两步各是一次短调用，固定 `--gpu-backend cpu`（独显留给主链） |

**上面 `ONNX_*` / `GPU_PY` / `GATE_*` 那几行只在 `PRIMARY_LEG="onnx"` 时被读到**（10-14 起
出厂默认是 crispasr）。驱动里那两段的标题也标了这一点；`should_switch_interpreter()`
和 `onnx_leg_setup()` 都按腿分支，默认腿下一次 crispasr 短调用都不发。
`LID_ROUTER` / `FRAME_LID_*` / `FFMPEG_EXE` 那几行正相反 —— 只在 `crispasr` 腿生效，
`onnx` 腿的语种闸门原样不动。**10-16 更正**：这一节原先写"默认腿下一行 ORT 都不 import"，
帧级路由上线之后**这句不成立了** —— `frame` 那趟要在 crispasr 之外建 FireRedLID 的 ORT 会话，
所以默认腿现在确实会 import onnxruntime；但它落在 **CPU**（`lid_onnx.py` 建会话时 providers
写死 `CPUExecutionProvider`，驱动没改那个文件），因此**不涉及** `GPU_PY` 换解释器那套，
系统 python 里只有 CPU EP 的那份 onnxruntime 就够用。`provider=cpu` / `--stop` / `--help`
这些短路径照旧一行 ORT 都不 import。

实测过的数（GPU，4060）：单文件 40 s 量级的中文素材 **RTF ≈ 0.08**。

## 5. 已知洞与边界

- **VAD 为什么不用 crispasr 默认的 silero v6.2.0**：v6 对唱歌素材判 0 段 → 整条静默丢弃
  （rc 仍为 0、不落 `.txt`）。同一条 60 s 日推歌曲副歌：`firered-vad` 出 97 字真歌词、
  silero-v5 捡回 59 字、silero-v6 出 0 段。**代价是多花约 31% 墙钟**。
  FireRedVAD 的 voice 类是"语音 ∪ 唱歌"，所以它不会把唱歌整段丢掉。
- **CrispASR 自带文档与实现不符的三处**见下面 5.1，读源码就能复算，不需要跑 GPU。

### 5.1 与 CrispASR 自带文档不符的三处（读 v0.8.37 源码核出来的）

三条都是**代码级**断言，复算方法：clone 上游后 `git checkout d08ec2d`（= 0.8.37，发布件
`crispasr.exe --version` 打的 git sha 就是它），照下面给的 `文件:行` 逐行读。等级 **A**
（本包内可复算的读码结论，不依赖任何一台机器）。

**① `PERFORMANCE.md` 里 P0 那行说 firered 解码器自注意力"没有 KV cache —— 增长向量、O(T²) 重算"，
这条不成立。**

* 每个 beam、每一层都带一份历史 K/V：`src/firered_asr.cpp:2247-2258` 声明 `sa_k` / `sa_v`
  并按 `max_len` 预留容量；`:2340-2343`（贪心路）与 `:2563-2568`（beam 路）每步只把**当前这一步**
  的 K/V `insert` 到末尾，历史部分一次都不重算。
* 交叉注意力那侧的编码器 K/V 也是每层算一次、全部步复用（`K_enc[li]` / `V_enc[li]`，`:2372-2395` 里只读不重算）。
* 随长度增长的是 attention **打分本身**：每步 O(T)、整句 O(T²)。这是注意力的固有代价，
  和"没做 KV cache"是两件事，P0 那行把它们混成一件了。
* 误判可能的来源：context 结构体里确实有 `kv_self_k` / `kv_self_v` / `kv_cross_k` / `kv_cross_v`
  四个张量字段（`src/firered_asr.cpp:254-259`），但**全文件除声明外没有任何使用点**，是死字段。
  "结构体里有 KV cache 字段却没接上"看起来正好是 P0 那条说法的样子，实际在用的缓存是上面
  那两份 per-beam vector。
* 上游自己已经推翻过这条，而且就在同一份发布件里：`HISTORY.md:3449-3458`（2026-07-12 那节标题
  直接写着 "profiling debunked the handover"）用 `FIRERED_BENCH` 逐节点量下来是——self-attn 的
  K/V **已经缓存**（`beam.sa_k/sa_v` 每步只 append 当前 token）、单步耗时**与历史长度无关**，
  所以解码是 **dispatch-bound** 而不是注意力复杂度问题，真实开销是每步 90 多个小 matvec
  （8 个投影 × 16 层）。他们的修法是常驻图缓存 `CRISPASR_FIRERED_MATVEC_CACHE`
  （`src/firered_asr.cpp:385-391`，**默认开**，关掉走原来的每次建图路径，转写逐字节相同）。
  也就是说 0.8.37 到手时那条 P0 早已不成立，`PERFORMANCE.md` 那张表只是没跟着改。
* 对本包的意义：**别**用"CrispASR 没治 KV cache，所以 `cpu` 分支那条纯 ONNX 链更快是治了这个"
  来解释速度差——两侧都是增量 K/V，这条解释已经作废（那半边也有公开更正记录）。10-08 同机同件
  （80.2 s 素材、两边都 12 线程、都是 VAD + AED + 标点的整链）实测：ONNX 单件 23.85 s（RTF 0.297，
  另加 6.4 s 一次性装载），crispasr 贪心档 wall 31.9 s（自报 0.357）、默认 beam=3 wall 43.9 s
  （自报 0.501）；但同日更早那轮"单件对单件、两边各含一次装载"是 29.2 s 对 29.4 s，**打平**。
  也就是说差距来自**批次摊薄**（crispasr 每个文件重装一次模型、多个 `-f` 又省不下来），
  过去那个跨机器拼出来的"5 倍"从来不存在。剩下的差要归到计算图与执行器上（f32 编码器 + int8
  解码器的 mixed 档，见 `cpu` 分支 `FireRed-ONNX/README.md` §4.3）。

**② session API 没法把 FireRed 的解码固定在贪心：`beam_size = 1` 被当成"不设"。**

* `src/crispasr_c_api.cpp:1832` 里 `s->beam_size` 的初值就是 1；`:7551-7553` 写的是
  `if (s->beam_size > 1) firered_asr_set_beam_size(ctx, s->beam_size)`。于是调用方给 1 = 不调 setter
  = 落回 firered 自己的默认，而 `firered_asr_context_default_params()` 给的默认是 **3**
  （`src/firered_asr.cpp:297`）。"要贪心"和"没意见"在这个接口上是同一个值。
* 连带的第二条：firered 的 F1 复读熔断只接在 `beam_size == 1` 那条分支上
  （`src/firered_asr.cpp:2274-2283` 的注释明写 beam 路什么都没有，并举了硬音频上 "OOH"×35、
  跑满 `max_len` 烧掉约 350 s 解码的例子）。走 session API 的调用方同时拿不到贪心**和**复读熔断。
* **CLI 是另一条路，`-bs 1` 能进贪心**：`examples/cli/crispasr_backend_firered_asr.cpp:47` 直接
  `cp.beam_size = params.beam_size > 0 ? params.beam_size : 3`，1 会被传到
  `src/firered_asr.cpp:2311` 的 `beam_size == 1` 分支。（一处自我更正：早期我把这条写成
  "CLI 永远跑 beam 3、`-bs 1` 走不到贪心"，**CLI 那半句是错的**，走不通的是 session API 那条。）
* 本包这份驱动走的是 CLI（`crispasr.exe` 子进程），命令里**没有** `-bs`，所以取 CLI 的默认档。
  10-08 校机实测已核到：**不给 `-bs` 时日志逐条是 `firered_asr: decoder starting (max_len=150, beam=3, layers=16)`
  ⇒ 默认档就是 3**；给 `-bs 1` 才变 `beam=1`。`--help` 里那行 `-bs N [greedy]` 与运行时不一致，以运行时为准。
  10-08 之后把"CLI 不给 `-bs` 时那个默认值到底是几"这条从推断升级成了读码（等级 A，`git show d08ec2d` 可复现）：
  CLI 的活参数表是 `examples/cli/whisper_params.h:31`，`int32_t beam_size = -1;`，行尾注释写着
  "-1 = greedy; beam search only when explicitly set via -bs N" —— **这句注释对 qwen3 成立（`-1 > 1` 为假 ⇒ 不进
  beam 分支），对 AED 不成立**，因为上面那条适配器三元式把 -1 折成了 3。另：`examples/cli/cli.cpp:69` 那份带
  `whisper_full_default_params(...)` 的参数表整体在 `#if 0` 死块里（`:56` 起），早先"CLI 默认取自 whisper 默认参数"
  那句是读错了活代码，作废。
  要知道自己拿到的是哪一档，就对着日志里 `firered_asr: decoder starting (max_len=…, beam=…, …)`
  那一行看（`src/firered_asr.cpp:2269-2271`，verbosity ≥ 1 就打 = 默认会打，加了 `-q` / 静默档就不打）。
  速度代价也实测了：beam=3 比 `-bs 1` 慢 35%（12 线程）～49%（18 线程），加线程救不回来；
  "beam=3 更准"这一条**没有实测**，只是先验，动它之前自己复验。
* **温度（`-tp/--temperature`）不是 AED 的旋钮**（A，读码）：`--help` 那行
  （`examples/cli/cli.cpp:1046-1047`，默认 0.00）是通用的，但 firered 适配器全文不读
  `params.temperature`，`src/firered_asr.cpp` 也没有这个符号，能力表里没置 `CAP_TEMPERATURE`
  （`examples/cli/crispasr_backend_firered_asr.cpp:36-37`）⇒ 对 AED 传 `-tp` 是被静默忽略。
  真读它的是 LLM 那类后端：`examples/cli/crispasr_backend_qwen3.cpp:370` 塞进解码配置，`:396-399` 只在
  `temperature > 0` 时才把 argmax 换成 `sample_temp`（`src/core/greedy_decode.h:114`：logits 除以 T
  → softmax → 按分布抽，配 `--seed`），`:374` 的 `n_runs = (T > 0 && best_of > 1) ? best_of : 1`
  说明 **qwen3 的"多条路径"是 best-of-N 独立采样再取累计分（`:429-430` 打 `best-of-N picked score=`），
  不是 beam**；beam 那路要 `-bs > 1` 才走（`:269-305`，`core_beam_decode`）。
  `--temperature-inc`（默认 0.2）全文只被 whisper 家族读（`examples/cli/cli.cpp:2997`、
  `examples/cli/crispasr_backend_crispasr.cpp:120`），AED 与 qwen3 都不认。

**③ 每个切片的解码上限是硬顶 150 token，没有任何 CLI 参数能改 —— 所以 `--chunk-seconds` 不是速度参数。**

* `src/firered_asr.cpp:2075`：`int max_len = is_lid ? 2 : std::min(T_sub, 150);` —— 150 写死在源码里。
  `-n/--max-new-tokens`（默认 512）是 LLM 后端的路径（`src/core/greedy_decode.h:71`、
  `src/core/beam_decode.h:112`、`examples/cli/crispasr_backend_qwen3.cpp:253`），firered 适配器全文
  不读它，所以对 AED 无效。
* 该文件 `:2072-2073` 的注释按"3-4 BPE token/s"估算，认为 150 够用。**中文连续旁白实测到
  5.4-5.5 token/s**，150 token ≈ 27 s，所以 30 s 的切片会顶到 150 并静默丢尾。10-08 同机同链
  只差这一个参数、跑两条素材（逐片 token 数取自 crispasr 自己的日志）：
  s1.wav 80.2 s —— `30` 切 3 片 token 150/150/118（两片撞顶），去标点 445 字；`20` 切 5 片
  token 109/100/109/108/10（零撞顶），去标点 **463 字，+18 字 / +4.0%**（原文 1367→1423 B）。
  s2.wav 344.9 s —— `30` 切 12 片 token 150/150/115/146/130/145/150/150/150/150/150/141
  （**12 片里 7 片撞顶**），去标点 1771 字；`20` 切 19 片最长 135，去标点 **1850 字，+79 字 / +4.5%**
  （原文 5542→5818 B）。丢的都是整句尾巴：s1 缺"遍地""定独自持枪突入在所有民众的镜头中""毙"，
  s2 找回来三个完整句子。撞顶时 rc 仍为 0、日志只打 token 数、不报截断。
  同轮 `20` 档重复跑一遍逐字复现（43.9 / 43.6 s，token 序列与字数相同）。
  **速度代价接近零**：beam 档 s1 44.0→43.9 s、s2 184.4→181.4 s，贪心档 s1 36.2→31.9 s（−11.9%）。
* 对照：`--vad-export-raw` 那条文档说 VAD 段是 chunk-length-independent 的，但 crispasr 的 firered VAD
  实现（`src/firered_vad.cpp:446` 起）只有 `min_speech_sec` / `min_silence_sec`，**没有** 上游 FireRedVAD
  那个 `max_speech_frame=2000`（= 20 s）的长段强切。也就是说在这份构建里，切片的唯一上限就是
  `--chunk-seconds`，它必须 ≤ 20 s 才落在 150 token 之下。
* 补一层机制（10-09 读码，A）：**不传 `--chunk-seconds` 时这个上限不是"30"，而是没有**。
  `examples/cli/crispasr_run.cpp:1058-1062` 见到 AED 的 `CAP_UNBOUNDED_INPUT` 就把
  `effective_chunk_seconds` 置 0，而 `examples/cli/crispasr_long_audio_fallback.h:57` 的
  `if (wants_vad) return false` 让那条 30 s 兜底在开了 `--vad` 时不触发 ⇒ VAD 段多长就整段喂多长
  （`:1153 → :1251` 把 0 原样传进 `crispasr_compute_audio_slices`）。所以这条链上"传不传 `-ck`"
  是内容完整性参数，不是速度参数；上游对同一件事的措辞见 `CrispASR-Qwen/README.md` §4.1 ④ 引的
  `HISTORY.md:6210-6216`。
* 本包的处置：`Flid-Cg-firered-Cg-qwen.py` 的 `VAD_MAX_SEGMENT_SEC` 由 30 改成 **20** ——
  在 10-14 起的出厂默认腿（crispasr）下这一条是**正确性前提**，不是可选项；只有 onnx 腿
  因为不撞 150 顶才可以放宽。
* 这个坑**只咬 AED**：qwen3 那条解码预算是 `max_new_tokens`（默认 512，`-n` 能改），
  30 s 档离顶还有 2~3 倍，所以 `CrispASR-Qwen/` 与 `cpu` 分支 `Qwen3/` 的 30 没动 ——
  推导与复核方法记在 `../CrispASR-Qwen/README.md`。

- **CrispASR 这份构建把 qwen3 模型内置的语种判别吃掉了**（`--list-backends` 没置
  `CAP_LANGUAGE_DETECT`），兜底那趟仍然靠 `-l auto` + 前置判别器（10-14 起那件是
  FireRedLID，不再是 whisper-tiny；`_base_cmd` 对两个引擎都挂同一对
  `--lid-backend/--lid-model`，`Flid-Cg-firered-Cg-qwen.py:1463-1466`）。
  把驱动判到的语种码再传一次会把筛子的错误固化，所以不传。
- `--stop` / STOP 文件是**批边界**退出，一批最长可能等 `BATCH_SIZE` × 单文件时长。
- MSVC 4 件与 CUDA 运行库 3 件：前者是启动门槛（不补 rc=127，实测过），后者缺了会退化成
  CPU 后端或直接报错。

### 5.2 10-14 把批内筛子换成 FireRedLID（`--lid-backend firered`）

换的理由只有一句：主链是 FireRedASR2-AED，而 FireRedLID 与它是**同一个编码器 + 同一个
解码器结构**、只是词表换成 120 个语种码（`src/firered_lid.cpp:3-6` 的头部注释），
和 VAD / Punc 一起凑齐 FireRedASR2S 那四件。旧筛子 ggml-tiny 是另一家的模型。

下面四条是从 v0.8.37 源码逐条读出来的（**A 级**，"读码得"而非"本机实测"），
每一条都是相对 whisper-tiny 那件的**退步或改变**，用之前要过一遍：

* **判的音频更短：只有头 5 秒。** crispasr 的外部判别器先把喂进判别的音频截到 15 s
  （`src/crispasr_lid.cpp:284`，`kLidMaxSamples = 16000*15`），FireRedLID 自己再截一刀
  （`src/firered_lid.cpp:78`，`constexpr int kMaxLidSamples = 16000 * 5`）⇒ 交集 5 s。
  这批素材的头 15 秒常常是前奏音乐，5 秒更容易整段落在音乐里；whisper-tiny 那件走
  同一对截断时是 15 s。**本语料召回未测**（§6.6 末条那组 head15/head30 对照是
  ggml-tiny 的数，换了判别器以后没有重测）。
  **而且这组旧数指的方向是"窗口越短越容易判错"**：同一批三件，15 s 窗口的批内筛子判出
  `km 0.411` / `zh` / `ko 0.405`（两件被踢去 qwen3，批次 14.0 min、产出 1/3），
  30 s 窗口的闸门判 `zh×3`（6.1 min、3/3）—— 见 §6.6。把头 5 秒当输入的 FireRedLID
  落在比 15 s 更短的那一侧，所以**方向上是不利的一档**，只是它换来了"和主链同族、
  方言码表齐全"。这一条是本改动最需要用真批次验一遍的地方（§6.7 第一笔）。
* **置信度恒 1.0。** `src/firered_lid.cpp:90` 写死 `*confidence = 1.0f;`，注释是
  "beam search returns argmax, no explicit probability" ⇒ 日志里那句 `p=1.000` 不代表
  有把握，也**没有**"拿不准就按主链/按兜底"这条余量可留。whisper 那件给的是真概率
  （旧筛子日志里见过 `p=0.41`、`p=0.405`）。
* **判别失败不会退回 whisper，而是钉成 `en`。** firered 分支失败只打
  `crispasr[lid]: firered LID failed` 就 return false（`src/crispasr_lid.cpp:196-235`，
  `out.source` 保持 `"firered"`），随后 `examples/cli/crispasr_run.cpp:1013-1021` 因为
  `-l auto` 没给具体码而 `params.language = "en"` 并打一行
  `crispasr: LID failed and no -l was set — defaulting to 'en'`。`en` 在 AED 的覆盖范围内
  ⇒ 文件按主链跑、**编造风险静默发生**，兜底不触发。驱动那侧 `_LID_NOVERDICT_RE`
  （`Flid-Cg-firered-Cg-qwen.py:1264`）认这行、只记日志不扣文件。
  silero 那条分支才有"退回下一个判别器"的行为，firered 这条件没有。
* **每件重装一次、且默认上显卡。** `firered_lid_init`（`src/firered_lid.cpp:49-54`）
  直接套 `firered_asr_context_default_params()`，那份默认里 `use_gpu=true`
  （`src/firered_asr.cpp:297`）；而 `examples/cli/crispasr_run.cpp:1029` 每处理完一个文件
  调一次 `crispasr_lid_free_cache()`（#35 的显存修补）。⇒ 这 **整份 LID 权重是每文件重装**
  （出厂档 f16 = 1.81 GB；8 GB 显存的机器取 q4_k = 543 MB），
  不是批量加载一次（同一批里模型本体和 VAD 确实只加载一次，但前置判别器不在那里面）。
  驱动另外还设了 `CRISPASR_GGUF_MMAP=0`。**这一笔的时间与显存代价本机未量过**，
  是"要不要换档/换回 whisper"的唯一缺项。

**同时修的一个会全盘误判的错**（这条是改动的必要条件，不是优化）：FireRedLID 的标签表
是模型目录里 `dict.txt` 的 120 项，中文在里面是 `zh` **加** `mandarin/xinan/min/wu/xiang/
north/yue` 这类方言名，实际输出可能是 `zh mandarin` 这样的**两个 token**（模型卡原文
"Output: Language code (e.g., "en", "zh", "mandarin", "ja")"）。旧代码
`code.lower().split("-")[0] in _AED_IN_RANGE` 按整串查，`zh mandarin` 不在集合里 ⇒
**每个中文文件都被判成范围外**、整棵树白跑一遍 qwen3。现在按空白/连字符切开取
**首个 token** 再查，并把六个方言名一并算在范围内（`yue` 本来就在）。
选"首 token"而不是"任一 token"是有方向的：判错成"范围内"= 主链跑完还能重跑兜底（可救），
判错成"范围外"= AED 已经拿汉字编造过（不可救），所以规则偏向多跑兜底那一侧。
这个谓词单独抽出来跑过正反例（`zh mandarin` / `yue` / `other` / `en` / `""` /
`km` / `"zh-Wu"`），但**没有跑过真批次的判别结果**。

换回 whisper 那件只需两行：`CRISPASR_LID_BACKEND = "whisper"`、
`CRISPASR_LID_MODEL = os.path.join(MODEL_DIR, "ggml-tiny.bin")`。两件都在 `assets.json`
里，不用补下载。

## 6. `"onnx"` 档：语种闸门 + FireRedASR2S 全链路（10-09 建的，10-14 起是可选腿）

### 6.1 形状

```
一个文件
  ├─ 闸门（crispasr 的两次短调用，固定 --gpu-backend cpu）
  │    1) --vad -vm <silero v6.2.0> --vad-export-raw <json> --strict-pipeline --require-vad
  │         → 真语音段表。0.46 s（80.2 s 素材）
  │    2) 按语音段拼前 15 s → 临时 wav → -m ggml-tiny.bin -dl
  │         → auto-detected language: xx (p = …)。1.07 s
  │    合计约 1.5 s/文件
  ├─ 判到 _AED_IN_RANGE 内 → ONNX 全链路（进程内常驻，不重载模型）
  │    FireRedVAD(ONNX) → AED(aed_ort.py, mixed 图) → FireRedPunc(ONNX)
  ├─ 判到范围外 → 不进主链，攒进 flagged → 同批换 crispasr qwen3 兜底（原逻辑不变）
  └─ 容器读不了（m4a / mp4 / aac，/libsndfile 不认）→ crispasr 的【同一个】主引擎模型
        它内部带 ffmpeg；不换语种、不换引擎，只换个解码器
```

改动只在"主力那一路由谁实现"。队列、硬链接暂存、批内实时结算、`no_speech.txt` 记账、
隔离区、熔断回搬、`.lock` 单实例、Job Object、`--start/--stop`、通知全部原样；
`_crispasr_batch` / `_crispasr_one` 一行没改（只是加了 `_` 前缀）。**10-14 起出厂默认就是
`"crispasr"`**，本节这套闸门与 ONNX 主链要 `ASR_PRIMARY_LEG=onnx` 才跑；两腿之间没有任何
数据面差异，切来切去不用搬文件。

### 6.2 闸门为什么是这两步，以及它的失手方式

* 第 1 步必须 `--vad-export-raw`：不带 `-raw` 导出来是 `kind="chunks"`（30 s 网格），
  不是语音段（10-09 实测）。段表里 `start/end` 是**采样点**、`t0_cs/t1_cs` 是**百分秒**，
  文件里不写单位；驱动取带 `_cs` 的那两个字段除 100 —— 名字自己说明单位，不靠猜。
* 第 1 步不带转写：给它 `-otxt -of` 也不落 `.txt`、不跑一个 token（rc=0 / 0.46 s /
  输出文件"无"，实测）。所以闸门不会"顺手先转一遍"。
* **两步合成一次调用不行**：`--vad-export-raw` 与 `-dl` 同时给时 crispasr 0.46 s 退出、
  rc 仍是 0，但**没有判别行** = 静默假成功（实测）。所以这里是两次进程调用。
* **silero 一条语音都没找到时，crispasr 自己不空手退出**：长素材（实测 120 s）会打一行
  `VAD returned no speech at all on a Ns clip — falling back to full-clip chunks`，然后
  **把整条按 30 s 网格当段表写出来**（`kind` 仍是 `vad_segments`，JSON 里没有任何标记能区分）；
  20 s 的纯器乐则老实地落 `slices: []`。所以段表"有段"不等于"有语音"。驱动按 stderr 那句
  （取纯 ASCII 片段 `full-clip chunks`，那行里的破折号过编码转换会花）把网格丢掉，让已经写好
  的 0 段分支接手 —— 两条分支的处置本来就相同（都判原始文件），但不丢掉的话日志会把
  "原始前 15 s"说成"语音前缀"，恰好与实际相反（10-09 校机实测，`tmp/c02.vad.json` 与
  `tmp/c04.vad.json` 两份段表都在）。
* 正对照（合成件 `c03_ctrl_music.wav` = 15 s 合成器乐 + 20 s 中文语音，标准答案 zh）：
  原始前 15 s → **en 0.784（错）**；语音前 15 s → **zh 0.966（对）**。探针
  `b1009_scan/gate5.py` 先测出这两个数，10-09 校机又用**出厂驱动的 `gate_language()` 本身**
  复跑了一遍（`sbh1009/probe2.py`），数值逐字相同 = 这条改进真的在跑，不是只在探针里成立。
  同一批 7 件里 VAD-first 只改判了这 1 件，另外 6 件两问一致，且**没有一件被改坏方向**
  （没有"原始判在内、闸门判到范围外"，也没有反向）。这就是第 1 步的全部价值 ——
  不加 VAD 的 `-dl` 会被片头音乐带偏（她记录里第 (6) 条那个老洞）。
* 判据沿用 `_AED_IN_RANGE` 那份名单，没有另立一套；ONNX 这份 AED 与 crispasr 那个是同一个模型，
  覆盖范围一样，而且**同样不吃语言标记**（判到的码只用来挑文件）。
* 闸门没有结论（抓不到判别行 / crispasr 非零 / 超时）→ **不扣文件，按主链跑**并记一行日志。
  宁可让 AED 编造一段（老风险），也不能因为闸门自己瞎了就把整棵树拖去慢 5 倍的兜底引擎。
  这一层的"瞎了"由起批前的 `gate_selfcheck()` 拦：拿前 3 个待处理文件实跑闸门，
  一个判别行都抓不到就 **rc=2 不启动**（`onnx_leg_setup()` 同样在起批前建一次三个会话，
  缺权重、provider 没落在卡上、依赖没装，都在这里当场停，不会偷偷改走 crispasr 跑完整夜）。

### 6.3 只用 N 卡，不碰核显

`ONNX_PROVIDER` 只认 `cpu` | `cuda`（出厂 `cuda`，10-09 从 `cpu` 改的；纯 CPU 的机器要显式
给 `FIREDASR_ONNX_PROVIDER=cpu`，否则会在起批前那道 `enc_ep` 断言上当场停），`ep_list()`
对其它值直接退出。理由是**装了什么包不等于
跑在什么设备上**，三条都是为此：

* **不列 DirectML / OpenVINO**：那两个可以挑中 AMD 核显（这台笔记本的 iGPU 是 Radeon 780M，
  `Win32_VideoController` 打的型号，10-09 读的；集显跑重内核还会整机崩），ONNX 主链里唯一的 python 侧入口就是这一个常量。
* **VAD 与 Punc 恒定 CPU**：驱动在建 FireRedVadOnnx 会话时临时替换 `InferenceSession`
  工厂，把 `providers` 钉成 `["CPUExecutionProvider"]`（vendor 的 `infer_onnx.py` 自己没给
  providers，装了 onnxruntime-gpu 时它会自己挑设备），顺带补 `intra_op_num_threads`
  —— 那个文件建会话时没设线程数，ORT 的 0 = 吃满所有物理核。Punc 是 4 亿参数的 BERT、
  每段只推一次，上卡只跟编码器抢显存。不改 vendor 文件本体。
* **建完会话读 `get_providers()` 验一次**：`ONNX_PROVIDER="cuda"` 而 `enc_ep` 里没有
  `CUDAExecutionProvider` 就抛错。CUDA/cuDNN 运行库不齐时 ORT 只打一行警告就把整个会话
  退回 CPU，不读实际结果就是"说好用 N 卡、实际在 CPU 上跑一整夜"而没人看得出来。
* **起批前换一个解释器（`GPU_PY`，10-12 加）**：同一台机器上两个解释器的读数实测差得很干脆 ——
  系统 python 里的 ORT 1.24.2 只列 `AzureExecutionProvider / CPUExecutionProvider`，另一个装了
  `onnxruntime-gpu` 1.30.0 的 venv 列 `TensorrtExecutionProvider / CUDAExecutionProvider /
  CPUExecutionProvider`（10-12 本机各跑一次 `get_available_providers()`）。上一条断言过不过，
  完全由"这份脚本被哪个 python 起了"决定，所以驱动在起批前自己处理：`GPU_PY` 非空 + 当前解释器
  没有 CUDA EP → `os.execv` 原地换成那个解释器重启一次（记账位 `FIREDASR_GPU_PY_SWITCHED=1`
  防循环，实测换完 `sys.executable` 就是目标解释器、再判一次返回不动）。三条边界：
  ①**换解释器不等于换硬件**，只解决"用错了 python"，CUDA/cuDNN DLL 不齐那种"编译列表里列得出
  CUDA 却起不动"不由这一步兜，还是 `enc_ep` 断言停；②Windows 的 `os.execv` 走 MSVCRT、给参数
  拼命令行时不加引号，所以解释器或脚本路径含空格时**不换**（停在断言上比"换了个 python 结果它说
  找不到脚本"好读）；③出厂 `GPU_PY` 是空串 —— 仓库里不写任何机器专属路径，谁在机器上跑谁用
  `FIREDASR_GPU_PY` 给，或像桌面副本那样把 default 写成那台机器的路径。

显存与解码器上不上卡：默认 `ONNX_DEC_ON_GPU = False`（编码器上卡、解码器留 CPU）。
10-09 在 N 卡上自己测过（§6.6）：**放不下不是理由** —— 编码器 + int8 解码器都上卡时峰值
4,788 MiB，8 GB 档放得下（只放编码器是 4,256 MiB）；不让解码器上卡是因为**它上卡更慢**
（同一批三件素材 wall 418.9 s → 537.2 s，慢 28%）。逐 token 自回归的解码器每步都要发一批
kernel launch，crispasr 那边同源结论是 per-token GPU launch 20 ms 对 CPU 一整步 60 ms
（§5.1），两边方向一致。

### 6.4 为什么"整段无语音"和"判到范围外"处置不同

* 闸门判 0 段（silero 对唱歌素材的已知形态）→ 拼不出语音前缀 → 退回对**原始文件**直接 `-dl`，
  照样有语种结论，不会因此扣文件。crispasr 把 0 段换成 30 s 整段网格的那一种也走这里（§6.2）。
* ONNX 链自己那步 FireRedVAD 判 0 段 → 转写是合法空串 → `settle()` 走 `no_speech.txt` 记账、
  源文件照删。与 crispasr 那条 `no speech detected in '…'` 的记账口径相同。
* 转写侧的 VAD **没有**换成默认 silero：v6 会把唱歌素材整条吞掉（§5 第一条）。
  silero 在这里只出现在闸门，因为闸门要的恰好是"人在说话的那 15 秒"，不是"整条内容都在"。
* 撞 cache 上限被截断的段数会随统计打出来（`st["trunc"]`），与 `cpu` 分支那个驱动同口径；
  ONNX 这条的 VAD 自带 20 s 强切（上游 `max_speech_frame=2000`），结构上到不了 §5.1 ③ 那个 150 token 顶。

### 6.5 校机 CPU 冒烟（10-09 已跑完）

沙箱 = 一台 24 逻辑核、无独显的机器，`provider=cpu`、`graph=mixed`、`threads=4`，7 件素材
每件打一条分支，与生产批次同机共存 —— 所以下面的耗时是**被抢占中的读数**，只会偏高不会偏低。

* 建链：VAD 0.1 s + AED（mixed 图，权重 3358.3 MB）5.7 s + Punc 1.7 s = **7.4 s**，
  一次建好整批常驻。
* 主链单件：80.23 s 音频 / 79.27 s 语音 → **30.3 s**（对语音 RTF 0.38）；
  120.0 s / 79.25 s 语音 → **23.8 s**（RTF 0.30）；7.81 s / 7.72 s → 2.25 s；
  20 s 纯器乐 / 0 段 → **0.04 s**（无语音基本不花钱）。
* 闸门：每件 **1.1–1.8 s**（silero 导段 + `-dl`，两次 crispasr 短调用）。
* 分支覆盖：zh/en 进 ONNX 主链；ja 不进主链、同批攒给 crispasr qwen3 兜底（120 s 那件
  兜底 42.8 s）；`.m4a` 被 libsndfile 拒了 → 交给 crispasr 的**同一个** aed 模型（它内部带
  ffmpeg）；20 s 纯器乐 → 主链 FireRedVAD 判 0 段 → 合法空串 → `no_speech.txt` 记账 + 源文件
  删除。整批 7/7 有结论，`rc=0`。
* 回归：三次运行的 `p.txt` **逐字节相同**（4917 B，sha256 前缀 `eef6ba05`），其中第一次运行
  闸门是坏的（下面第 3 条）—— 印证闸门只改**路由**、不改主链出字。c05 与 `cpu` 分支那个独立
  链路驱动在同一件上的输出，去空白、去标点后 **463 / 463** 字符一致，再归一大小写就逐字相等
  （唯一的差是 Punc 把英文代词 `i'm` 写成 `I'm`，而参照 JSON 存的是加标点**之前**的分段文本）。

改造加这一轮冒烟一共逮到四个缺陷，全部已修，而且**都在新增代码里**（`_crispasr_batch` /
`_crispasr_one` 一行没动）：

1. **缩进错误**：合并兜底分支时把一处 `if/else` 改坏了，文件 import 就 `IndentationError`。
   先前"这份已经能编译"的说法是**错的，在此公开更正** —— 当时只看了 diff，没跑 `py_compile`。
2. **`FIREDASR_ONNX_ASR_DIR` 的空值哨兵被吃掉**：`resolve_dir(name, "")` 把 `""` 拼成了本包
   目录，链路于是去包目录下找 `encoder.f32.onnx`，起链当场 `FileNotFoundError`。这个空值的
   含义是"在 `ONNX_MODELS_DIR` 下按 `sherpa-onnx-fire-red-asr2*` 自动找"，现在只有环境变量
   真给了值才展开。现象是**起批前 `rc=2` 不启动**，不是偷偷换腿 —— 那道门禁本来就是为了
   这个才立的。
3. **闸门漏 `import json`**：`_gate_vad_segments` 读段表用 `json`，顶部却没 import，于是每件
   都抛 `name 'json' is not defined`、被 `except` 吞成一行 WARNING。闸门**看着在工作**
   （照样吐出 zh/ja 结论），实际全程走的是"判原始文件"那条退路 —— 功能没坏、改进没生效、
   日志只有 7 行 warning，是最阴的一种。修法是补 import，并把 7 件素材的两种问法并排打出来
   （`sbh1009/probe2.py`，就是上面那条"只改判 1/7"的表）来确认 VAD-first 真的在跑。
4. **`原始前 15s 那段没用上` 这句在单长段素材上是假的**（c05 的段表只有 0.03–79.55 s 一条，
   拼出来的前缀就是原始的前 15 s，一秒都没跳过）。日志改成给数：
   `语音前缀 15.0s(取自原始 0.0–15.0s 的 1 段)`；`_speech_prefix` 因此多返回
   `{head, tail, k}`。改完 `pyflakes` 在这份文件上只剩 `np` 那四处 —— 那是运行时从链路模块
   取的（`chain_module()` 里 `globals()["np"] = chain.np`），不是漏 import。

### 6.6 N 卡实机（10-09 已跑完，笔记本 RTX 4060 Laptop）

沙箱 = 作者自己的笔记本：独显 **RTX 4060 Laptop（`nvidia-smi` 报 8,188 MiB）**、核显
**Radeon 780M**（型号取自 `Win32_VideoController`，10-09 读的 —— 先前本节写的是 680M，
**那是错的，在此公开更正**）。环境 = 单独一个 venv（CPython 3.13.11，不带系统
site-packages），里面装 `onnxruntime-gpu==1.30.0`；系统侧是 CUDA toolkit 13.2 加
System32 里的 `cudnn64_9`。链路代码与三件权重从校机经 sftp 原样搬来，>1 MB 的 8 件
逐件 sha256 与校机那份对上（`mismatches: 0`）。

跑的件是**桌面那一份驱动**（与仓库这份同代码，只有 CONFIG 里的路径写法和本节这类说明
文字不同），输入是生产库里三件真素材的**副本**（66.92 + 296.86 + 617.78 = **981.56 s**
音频），输出与暂存都在沙箱目录，生产树一个字没动。

先单验运行时（`python -c "import onnxruntime"` + 三个会话各建一次）：1.30.0 在这台机器上
import 干净、`CUDAExecutionProvider` 在列、3.1 GB 的 f32 编码器会话确实落在卡上 ——
校机 10-05 那个"1.21 以上 import 就崩"的形态**没有复现**，所以那条是"看机器"，不是
"跨版本就不能用"（`requirements.txt` 最后那段已按这个改）。

四趟，同一份代码只换开关（g3 那份是把 `ONNX_DEC_ON_GPU` 改成 `True` 的副本，其余逐字相同），
输入是同一批副本：

| 趟 | 腿 | provider | 解码器 | rc | 整批 wall | RTF | 三件逐件用时 | `nvidia-smi` 峰值 |
|---|---|---|---|---|---|---|---|---|
| g1 | onnx | cuda | 留 CPU | 0 | **418.9 s** | **0.427** | 17.32 / 80.01 / 244.48 s | util 99%、4,256 MiB |
| g2 | onnx | cpu | 留 CPU | 0 | 555.4 s | 0.566 | 21.63 / 129.59 / 358.69 s | util 0%、210 MiB |
| g3 | onnx | cuda | **上卡** | 0 | 537.2 s | 0.547 | 18.98 / 136.67 / 336.26 s | util 100%、4,788 MiB |
| g4 | crispasr（参照） | `--gpu-backend cuda` | —（它没有这个开关） | 0 | 1084.2 s | 1.105 | AED 段 840 s 只出 1 件 / 兜底段 242 s 出 2 件 | util 峰值 96%、4,104 MiB（**这峰不是 AED 段造成的，见 g5**） |
| g5 | crispasr AED 腿单件定位 | `--gpu-backend cuda` | — | 0 | 253.4 s | 0.853 | `[魅族17]` 一件 296.86 s | util 中位 **0%**、p95 9%、单点最高 50%；显存恒 1,116 MiB |

* **编码器上卡有效，比纯 CPU 快 1.32 倍**（wall 555.4 → 418.9 s；逐件 1.25×/1.62×/1.47×）。
  `执行提供者:enc=['CUDAExecutionProvider', 'CPUExecutionProvider'] dec=['CPUExecutionProvider']
  punc=['CPUExecutionProvider']（VAD 恒定 CPU）` 这一行是驱动自己读的 `get_providers()`，
  不是推断。装载：VAD 0.1 s + AED 12.6 s + Punc 4.2 s = 17.0 s（整批常驻，一次）。
* **解码器上卡是负收益**：g3 比 g1 慢 28%，而且三件逐件都慢（1.10×/1.71×/1.38×）。
  显存反而不是瓶颈 —— 两个都上卡峰值 4,788 MiB，8 GB 档放得下（只放编码器 4,256 MiB）。
  所以 `ONNX_DEC_ON_GPU` 保持 `False` 的理由是**速度**，先前写"放不下"那条已经改掉（§6.3）。
  crispasr 那边同源的结论（per-token GPU launch 20 ms 对 CPU 一整步 60 ms）方向一致。
* **同机整腿对照（g4）**：同一批三件副本走 `PRIMARY_LEG="crispasr"` 是 **1084.2 s（RTF 1.105）**，
  ONNX 腿是 418.9 s（0.427）⇒ **ONNX 这条腿在这台机器上少 61%**。拆开看这 1084.2 s 花在哪：
  AED 段 11:46:41→12:00:42 共 **840 s 只产出 1 件**（三件的前置 LID 都在这一段跑，其中两件被判到
  范围外、扣下不转写），兜底段 12:00:42→12:04:45 共 **242 s 出 2 件**（684.7 s 音频 ⇒ 那一段 RTF 0.354）。
  口径要讲清：这是**腿对腿**的账（两边都把 3 件输入变成 3 件产出），不是"AED 对 AED"—— g4 里
  只有 1 件真由 AED 转写，另 2 件是 qwen3。crispasr 每个文件重装模型、批次 `-f` 摊不开那件事
  （`cpu` 分支 §12 的校机实测）在这台机器上同样成立，而且它这里还多了一层"LID 判完再换引擎重跑"。
* **g4 那个 util 96%/4,104 MiB 不属于 AED 腿 —— g5 单独定位过**（这是本节的第二条更正：
  我先前只留峰值、没时间戳，读起来像"AED 腿也在用卡"，和 §6.6 之外那份"显卡整天空闲"的旧读数冲突）。
  g5 = 只放 g4 里判到 `zh`、确实走了 AED 的那一件（296.86 s），`PRIMARY_LEG="crispasr"`，
  采样器每 2 s 取一次并记时刻（`gpu1009/rungpu_aed1.py` → `sb2/run/g5_samples.json`）：
  **120 次采样里 util>5% 只有 6 次**（离散在 t=2.3/61.9/76.8/143/183.4/236.4 s），单点最高 50%、
  中位 0%、p95 9%，显存全程 1,116 MiB 常驻。⇒ **AED 腿确实只把切片编码器零星丢给卡、解码在 CPU**，
  那条"整卡 2-6 W、显存占着 1,230 MiB"的旧读数与这次对得上；而 4,104 MiB 只能是 qwen3 兜底段
  （1.7B q8 权重上卡）留下的。RTF 口径也顺手提一句：g5 单件 253.4 s（0.853）比 g4 里 AED 段的
  840 s/1 件便宜得多，因为那 840 s 装了**三件**的前置 LID。
* **换腿会改文本，不只是改速度**（g1 对 g4，尺子＝去空白 + Unicode `P*`/`S*`）：
  唯一两边都由 AED 转写的那件（`[魅族17]`）**1,454 字 对 1,458 字、相似 0.9657**，逐处差异是
  近音近形摆动加一类系统性差别 —— **crispasr 保留英文大写（`FLYME`/`NFC`/`IPHONE的X`/`MX`/`OS`），
  ONNX 这条链出小写并与中文黏住**（`flyme`/`nfc`…，与校机 10-08 那条 B 级观察同向）。
  另两件在两条腿上根本不是同一个引擎（ONNX 腿判 zh 全进主链；crispasr 腿判 km/ko 转 qwen3），
  所以只有一处值得记：数字形态 qwen3 给 `2015`/`31`/`220V`/`5C`，ONNX 给"二零一五""二百二十伏"，
  且 qwen3 会把短句压掉（`sb模块走功率十五瓦` → `SB15W`）。唱歌那件 0.164，不参与判据。
* **核显一次都没被选中**，判据是机制：`ep_list()` 只给 `cpu`/`cuda`（DirectML、OpenVINO 不列），
  VAD 会话被钉成 `["CPUExecutionProvider"]`，而 `nvidia-smi` 只看得到 NVIDIA 设备 —— 上面那几个
  util/显存读数按构造就是 4060 的。g2 是正对照（util 0%、只有 210 MiB 上下文）。核显侧的占用
  事后没有读数，所以"780M 全程没干活"是**机制推断**，不是仪表证据。
* **功率那一列不采信**：这台机器的 `nvidia-smi` 里 `power.limit` 是 `[N/A]`，util 同刻 99–100%
  时 `power.draw` 最大只读到 23–24 W，显然不是真值。
* **文本层面 GPU 与 CPU 两趟几乎同字**：ONNX 那三趟里两件中文素材**字数完全相同**
  （1,454 / 3,365），两两相似度 ≥0.9988，逐处差异都是单字级的近音/近形摆动（继↔即、拓↔踏、
  的↔了、业↔叶、安↔n），没有丢句、没有多出整段。第三件是唱歌素材，两腿都把它编成英文歌词且彼此相距很远
  （相似 0.10–0.27），这一件不能当 EP 之间的判据 —— §5 第一条说的就是它。
* **闸门那句"silero 判 0 段"是 silero 的真实判定，不是报错被吞**：三趟日志里
  `闸门 VAD 失败` 出现 **0** 次；拿同一个文件的 ASCII 名副本单独复跑那条 VAD 调用，段表
  照常落盘、`kind="vad_segments"`、`num_slices: 0`。同一条素材主链那步 FireRedVAD 出
  43.16 s 语音 / 5 段 —— 正好是"FireRed 的 voice = 语音 ∪ 唱歌、silero 会把唱歌整段吞掉"
  在真素材上的一个实例（§5 第一条、§6.4）。
  顺带一条**探针自己的坑**（不是驱动的洞）：crispasr 是 ANSI 程序，`-f` 直接给中文名一律
  `error: input file not found`（rc=2，`-dl` 也一样），所以自写探针必须走驱动那套
  `staged_inputs()` 的 ASCII 暂存再去抄命令，否则测出来的"闸门坏了"是假的。
* **两腿的语种筛子判得不一样，原因已经测到底**（**这组数是 10-09 那版的**：当时批内筛子挂的是
  `--lid-backend whisper` + `ggml-tiny.bin`。10-14 起筛子换成了 FireRedLID，现役默认腿的判别器
  已经不是这一组里的"筛子"那一列，窗口结论照旧成立、件级结论要重取，见 §5.2 与 §6.7。
  这一条替换先前那条"未解释，B 级观察"，
  并且**更正我写错的两处机制**：老句子里"两次都是 `-dl`"不成立 —— 批内筛子不是 `-dl`；
  "差别在喂进去的音频范围与执行后端"也只对了一半）：
  同一批三件，ONNX 腿的闸门判 `zh×3`（全进主链，批次 6.1 min、产出 3/3）；crispasr 腿自带的
  前置筛子逐件判出 `km p=0.411`（`PPT动画…`）/ `zh`（`[魅族17]…`）/ `ko p=0.405`
  （`一家卖小米的粗粮公司…`，617.78 s）—— 于是 14.0 min 只产出 1/3，
  两件转 qwen3。入口是**两条腿读的根本不是同一行**。crispasr 自己的测试矩阵
  （`tools/test-all-backends.py:1269-1272`）把语种结论的打印形态列成四种，闸门读的是
  whisper 后端那一行 `auto-detected language: xx (p = …)`，批内筛子读的是框架前置步那一行
  `crispasr: LID -> language = 'xx' (whisper, p=…)`；后者走 `src/crispasr_lid.cpp`，那里
  `:284` 把样本写死截成 `16000*15`（**只看原始前 15 秒**），前者实测看的是**前 30 秒**。
  同一个 `ggml-tiny.bin`，差别只在窗口长度。

  窗口实测（探针 `b1009_lidprobe/probe.py` 一次跑完 P1–P13 + W1–W4，原始读数在同目录
  `probe_out.txt`；切片由脚本自己 `ffmpeg -t 15 / -t 30` 现截，命令与闸门逐字同形：
  `-m ggml-tiny.bin -t 2 --gpu-backend cpu -f <ASCII 名> -dl`）：

  | 素材 | 原始前 15 s | 原始前 30 s | 整件 | 批内筛子实际给的 |
  |---|---|---|---|---|
  | `PPT动画…`（66.92 s，silero 判 0 段 ⇒ 闸门退回判**原始件**） | `km p=0.410626` | `zh p=0.215178` | `zh p=0.215164` | `km p=0.411` |
  | `一家卖小米的粗粮公司…`（617.78 s，有语音段 ⇒ 闸门判 15 s 语音前缀） | `ko p=0.404644` | `zh p=0.993594` | 未测 | `ko p=0.405` |

  这两个"对上"不是巧合：筛子读的窗口就是我截的那 15 秒，所以它们该是同一个数（对上时按
  `%.3f` 舍入：0.410626 → `p=0.411`、0.404644 → `p=0.405`）。结论两句话：
  **这批 B 站素材的前 15 秒是片头音乐/音效，人声在第 15–30 秒才进来**，所以 15 秒窗口给出
  km/ko、30 秒窗口给出 zh；而 PPT 那件两条腿判的**都是原始文件**（闸门因为 0 段退回了原始件），
  它那一处的分歧**只有窗口长度**这一个变量。"30 秒"这个数不是我读到的常数，是从
  "head30 与整件读数相同到 1e-5、head15 是另一个码"倒推出来的。

* **还有一个更硬的实测：同一个文件在一条命令里可以给出三个不同的码。** `-dl` 加
  `--strict-pipeline` 时 crispasr 按切片逐段调后端，而 whisper 那个检测是**每次调用重做一遍**，
  于是 66.92 s 那件一次吐出 `zh 0.215164` / `en 0.459863` / `ko 0.441314`（三个切片各判各的）；
  换 firered VAD + `-ck 20` 变成 `zh 0.197991` / `en 0.456393` / `en 0.271964`；连跑三次逐字相同
  （P8 = P11 = P13，是确定性的，不是抖动）。同一条命令里 `-dl` + silero VAD 则**一行都不出**
  （silero 对这件判 0 段 ⇒ 没有切片 ⇒ 静默，`--strict-pipeline` 下也照样 0 行）。
  ⇒ "一个文件一个语种"不是 crispasr 承诺的性质，谁把它的判别行当逐文件的唯一结论用，
  都得先确认自己喂的是哪一种命令形状。
* **上面那条不影响 `_lid_map` 的对号**：它只匹配 `LID -> language = '…'`
  （`Flid-Cg-firered-Cg-qwen.py:1259` 那个 `_LID_RE`），而这一行 crispasr 每文件只打一次 —— 两处打印点
  `examples/cli/crispasr_run.cpp:986` 与 `:1003` 由 `probed_ok` 互斥，严格管线那趟在 `:5059` 同样只一次；
  `auto-detected language:` 那些逐段行根本进不了它的队列，所以"逐段三个码"打不乱顺序对号。
  g4 实测 3 件出 3 个事件，没触发"整批不采信"。残留风险按机制写在这：真出现"一件两行 `LID ->`"
  会撞 `len(events) > len(queue)` ⇒ **整批不采信**（失败方位是安全的，等于这次没兜底）；
  会错配的只有"一部分文件多打、另一部分少打"这种**部分超额**，那种形态到现在一次都没观察到。
* **小红书那把 VAD 认不认得出片头是音乐？** 10-10 补测（探针 `_tmp_toolchain/b1009_vadseg/vadseg.py`，
  只跑链里 VAD 那一环，会话在链里恒定 CPU；读数同目录 `vadseg_lidprobe.txt`）：同一条 66.92 s 素材，
  **FireRedVAD 出 5 段、语音合计 43.16 s** —— 与 g1 批次日志那句"音频 66.92s / 语音 43.16s / 5 段"
  逐字对上，所以探针量的就是链里那一把。段表（秒）：
  `4.28–5.17` / `16.74–19.07` / `26.96–40.78` / `40.79–58.94` / `58.95–66.92`。
  ⇒ **前 15 秒里它只留下 0.89 秒**（4.28–5.17），另外 14.11 秒判成非语音：**片头是音乐这件事
  FireRedVAD 基本认得出来**；在这件素材上判 0 段的是闸门用的那把 silero v6（§6.5），不是小红书自家那把。
  留下的那 0.89 秒是什么内容没核（要起一次 AED 会话才看得见，约 3.4 GB / 十几秒加载，标 B）。
  机制后果顺带记一句：silero 判 0 段 ⇒ 闸门拼不出语音前缀、只能退回判原始件；换成 FireRedVAD 那把的话，
  15 秒语音前缀会从 16.74 s 起拼（跳过片头）。**这条只写两个读数的关系，不等于要换** ——
  闸门用 silero 是她 10-09 定的（"先用默认的 silero vad"），换不换由她再定。
* 实践含义不变，而且更硬：换主腿之后**有多少文件进主链这件事会变**，不是纯换后端。
  反向也说一句：crispasr 那把只看原始前 15 秒的尺子，会**系统性地把片头音乐当语种证据**，
  这正是闸门第 1 步（VAD 拼语音前缀）存在的理由；现在多了一条 —— 光拼语音还不够，
  窗口短到只剩片头照样翻车。`GATE_LID_SEC` 提到 30 这条**她 10-10 已定：不调，保持 15**（§6.7）。

### 6.7 还没做的事

* **10-14 新筛子（FireRedLID）欠的两笔**（机制侧的账在 §5.2，这里只列没量过的）：
  ①每文件重装整份 LID 权重（f16 1.81 GB / q4_k 543 MB）且 `use_gpu=true` 带来的**时间与显存代价**——默认腿下一趟批里它占多少
  墙钟，没测；②它在**本语料上的召回**——旧筛子那组 head15/head30 对照（§6.6 末条）是
  ggml-tiny 的数，判别器换掉以后没重测，而它只看头 5 秒。
  这两笔是"要不要换档（q4_k/q8_0/q2_k）或换回 whisper"的全部依据缺口。
  **10-16：出厂档已经从 q4_k 换成 f16，换的依据是上游默认口径 + 量化在这条路上拿不到 repack 收益
  （§3 那一行），不是这两笔量出来的数 —— 两笔仍然欠着。**
  **10-16 更新**：出厂默认已经改成 `LID_ROUTER="frame"`，而帧级路由那条腿**根本不挂批内筛子**
  （`_crispasr_batch(..., lid_off=True)`），所以 ① 在默认配置下是一个不会再走的路径 —— 它只在
  降级成 `sieve` 或显式 `ASR_LID_ROUTER=sieve` 时才又是现役。② **仍然全量欠着**，而且帧级
  没有把它补上：路由改的是【看到多少音频】，不是【判得准不准】，判成 zh 的外语文件照样被 AED
  拿汉字编造并正常落盘。路由自己那笔新账（每文件一次整条编码的秒数）也没量过 —— 驱动逐文件在
  日志里打 `vad X.XXs + ffmpeg X.XXs + lid X.XXs`，第一批跑完就有数。
* `crispasr` 腿在这台笔记本上只量了整腿（g4）和 AED 腿单件的 GPU 占用（g5）；
  **beam 档、qwen3 兜底腿单独的速度都还没在这台机器上量过**。
* `ONNX_MODE` 仍是 `greedy`：beam 在 ONNX 链里实现了但未调通（慢 9 倍、输出退化）。
* ~~`GATE_LID_SEC` 是不是该从 15 提到 30~~ —— **10-10 她已定：不调整，保持 15。这条已收口，
  别再重开。** 记录一下当时的问题位（不重做、只留档）：闸门读的那条 whisper 检测实测吃前 30 秒
  （§6.6），所以拼 15 秒语音并没有把这个窗口用满；提上去的代价是确定的（每件判别要多读 15 秒
  音频，而闸门已经是每文件 1.1–1.8 s 里的大头），收益没有实测。
* 分语种验证集仍没有（与 `cpu` 分支同一条洞）。
* Linux 侧只到机制，没有实机跑过。

## 7. 帧级语种路由（10-16，`LID_ROUTER="frame"`，出厂默认）

她的原话是判据：**"不是检出多语就换 Qwen，而是检出 firered 不支持的才换……crisper 的 lid
能对整个文件跑吗，不能的话就用 onnx 吧，这样才能确保整个文件都是 firered 支持或不支持的，
日语或其他小红书不支持的语言占比要到起码三段才切换 Qwen，以防误判。"** 下面每一条都对着一句。

### 7.1 为什么必须走出 crispasr

"能不能对整个文件跑"这个问题的答案是**不能**，三条都读到了源码级：

* `src/crispasr_lid.cpp:284` —— 外部判别器入口把输入截到 `kLidMaxSamples = 16000 * 15`；
* `src/firered_lid.cpp:78` —— FireRed 适配器在里面**再截到** `kMaxLidSamples = 16000 * 5`，
  注释自己写了理由："The encoder is O(T²) per layer, so halving T gives 4x speedup"；
* `examples/cli/cli.cpp:496-500` —— CLI 只有 `--lid-backend` / `--lid-model` /
  `--lid-on-transcript` 三个语种旋钮，**没有任何窗口/时长参数**。

所以"整个文件都在 FireRed 支持之列"这句话，crispasr 自己给不出来；能给的只有那条
`onnx/optional-lid/lid_onnx.py`（她自己导的 FireRedLID ONNX），它按滑窗逐窗解码、不截断。
这就是路由为什么落在 crispasr **之外**：先判完再决定喂不喂 AED，而不是像筛子那样在进程里
判、拦不住已经开跑的转写。

`p=` 恒为 1.0（`src/firered_lid.cpp:90` 写死，那行注释是 beam search 取 argmax、不产出概率），
**结构上没有置信度闸门可加** —— 所以"以防误判"只能做成计段：`FRAME_LID_MIN_BAD_SEGS = 3`。

### 7.2 形状：四步，后两步复用 `asr_chain.run_lid`

`frame_route()`（驱动里那一支）对每个文件做：

1. **FireRedVAD 导段** —— 用 `CRISPASR_VAD_MODEL`，**不是**语种闸门那份 silero。理由硬：
   silero v6.2.0 对唱歌素材判 0 段（数字依据在 CONFIG · VAD 那组），而唱歌正是最需要换引擎的
   那片；FireRedVAD 与主引擎同源，分出来的段就是 AED 真正会吃到的那套分段。
2. **读成链路口径的 PCM** —— `_gate_load_audio()`（libsndfile）失手就交给 `_ffmpeg_decode()`。
3. **切子段 + 打包分块** —— `_split_units()` 把每条 VAD 段切成 ≤`FRAME_LID_UNIT_SEC`（6 s）互不
   重叠的子段，短于 1 s 的尾巴丢掉；`_run_lid_blocks()` 再把相邻子段打包成 ≤`FRAME_LID_SLICE_SEC`
   （60 s）的块，**每块各自编码一次**，返回的时间戳一律加回块起点还原成绝对秒。
   为什么要切子段：一条十分钟的日语独白在 VAD 下常常只有一两段（10-09 校机实测 80.2 s 素材的
   段表只有一条 `0.03–79.55 s`，记在 `_speech_prefix`），不切的话"≥3 段"永远达不成。
   为什么要分块：编码器是 Conformer 全自注意力，分数矩阵随时长**平方**涨（十分钟素材
   T=15000 帧 @25 帧/s，单层 4 头 ≈ 4×15000²×4 B ≈ 3.6 GB），夜间无人值守会被它撑死；
   分块后总代价 ∝ ΣTᵢ² 比整条**更小**，只有块边界上的窗少看几十秒上下文，而 3 s 窗判语种用不着。
4. **滑窗解码 + 按子段汇票** —— 直接调 `chain.run_lid(..., win_s=3.0, hop_s=1.5, mode="slice")`，
   与单文件跑 `--lid` 时同一条实现，**不复制一份代码**。

判据只有一条：`_in_aed_range()` 判到范围外的**子段数** ≥ `FRAME_LID_MIN_BAD_SEGS`。
范围 = 中文（含 `zh mandarin` / `zh xinan` / `zh min` / `zh wu` / `zh xiang` / `zh yue` /
`zh north` 这些方言二词）/ 英语 / 粤语；词表里那个真实的 `"other"` 类和任何范围外码都算坏票。

日志每文件一行，长这样（能直接读出这一条判据的证据链）：

```
80s 音频 / VAD 1 段 → 子段 14、窗 14(slice 3s/1.5s,每块≤60s,1 块) → 有票 14、范围外 14 [ja×14] 整条=ja | vad 0.31s + lid 1.42s
```

### 7.3 三种"不换引擎"是三件不同的事，必须分得开

* 有票、范围外子段不够 3 ⇒ **放行**（这就是她那句"不是检出多语就换"）。
* `votes=0`（VAD 零段 / 两种解码器都读不了 / 子段全无窗覆盖 / 会话没起来）⇒ **不路由**。
  "看不见"不等于"范围外"，与语种闸门那条同一口径；否则路由自己一坏就把整棵树拖去兜底引擎
  —— 那是比编造更难查的坏：一夜文件全换了引擎，而判据一条都没有。
* 判成 zh 的外语文件 ⇒ **放行，且这是已知假阴性**。帧级改的是【看到多少音频】，不是
  【判得准不准】；这一层没有任何东西收得住。

放行的那部分走 `_crispasr_batch(..., lid_off=True)`：**主链那趟不挂前置判别器**，顺带省掉
`examples/cli/crispasr_run.cpp:1029` 那个每文件重装整份 LID 权重（f16 1.81 GB）的开销（`crispasr_lid_free_cache()`
是 #35 显存 OOM 的修补，效果就是缓存在批内每个文件都要重建一次）。被路由扣下的文件**根本不喂
AED**，攒成一批换 qwen3 重跑，`process_batch` 的形状不用改（`results` 里 = 已产出，`flagged`
里 = 待兜底，两处都没有 = 逐文件重跑）。

### 7.4 起批前的降级门禁

`frame_router_setup()` 返回 0 = 按配置走、1 = **整条路由降级成批内筛子**、2 = 致命（配置本身不对）。
查的是"能不能建成会话"，不是"文件名在不在"——external data 缺件、ORT 版本不认这张图、
`kaldi_native_fbank` 没装，只有真建一次才看得见。三道关：

| 缺什么 | 后果 | 为什么这样处置 |
|---|---|---|
| `LID_ROUTER` 写了 `frame`/`sieve`/`off` 之外的值 | rc 2，不起批 | 拼错的字符串静默退回默认等于悄悄换了一条路由 |
| 起批前点验的四条路径缺任一件（`lid_encoder.onnx` / `lid_decoder_step.onnx` / `dict.txt` / `cmvn.ark`），或 VAD 不在，或会话建不起来 | rc 1 → 降级 `sieve`，critical 写清缺哪几件与补法 | 这些都是**建造产物**（`onnx/optional-lid/lid_export.py`，要 torch + FireRedASR2S 仓库 + 官方 `model.pth.tar`），不在任何下载清单里。两张图连各自的 `.onnx.data` 合计 3,552,376,747 B，校机 `models` 下已有整套，**笔记本上没有**。那两个 `.data` 大文件**不在点验清单里** —— 缺它们时文件路径全在、只有真建一次会话才看得见，这正是 `frame_router_setup()` 走"建会话"而不是"查文件名"的原因 |
| `ffmpeg` 找不到或跑不起来 | rc 1 → 降级 `sieve` | 见下 |

ffmpeg 那道关为什么是**整条降级**而不是"这三种容器跳过"：能读的走帧级、读不了的退回筛子，
需要在同一批里开两个 crispasr 进程（`--lid-backend` 挂在命令行上，不是按文件给的）；而"这台
机器有没有能用的 ffmpeg"是**环境属性**，不是单个文件的属性 —— 起批前一次判清、整批只走一条路、
日志只有一种形状。降级成筛子 = 退回 10-16 之前的行为，不是雪上加霜；留着不管才是给 m4a/mp4/aac
开一个**连筛子都没有**的洞，而小红书下载下来的封装恰好主要是这一类。

`frame_selfcheck()` 再拿队列头几个文件实跑一次（`GATE_SELFCHECK_FILES`）。这层不能省：路由坏了
**不产生任何错误** —— crispasr 照常返回 0、AED 照常出字，只是每个文件都"没证据 ⇒ 不路由"，
整批外语素材被编成汉字并正常落盘（与 `--no-prints` 把筛子搞瞎那个坑同源）。但它**不判致命**，
只 warning：唱歌素材 VAD 判 0 段是这类语料的已知合法形态，头几个文件恰好全这种素材不等于路由坏了，
拿它停机是把正常素材当故障。

### 7.5 还没量的与残留盲区

* **每文件的代价没量过**。路由多花的是整条音频的一次编码（跑在 CPU，`lid_onnx.py` 建会话时
  providers 写死 `CPUExecutionProvider`，驱动没改那个文件，所以不涉及 `GPU_PY` 那套换解释器）。
  日志逐文件打 `vad + ffmpeg + lid` 秒数，批末再打 `s/文件`，第一批跑完就有数。
* **显存无关，RAM 有关**：会话常驻 = 编码器 + 解码器两张图的 external data 一次读进本进程。
  换来的是不再有那个"每文件重装整份 LID 权重"（出厂档 f16 = 1.81 GB），且这趟跑在 crispasr 之外，主模型的显存分配与它无关。
* **残留盲区只剩一条**：libsndfile 与 ffmpeg **都**解不了的音频 —— 它会照旧喂 AED 且**不挂筛子**
  （`_ffmpeg_decode` 为它单独打 warning，名字和退出码留得下）。这是 `frame_route()` 里唯一
  一处"整条判不了"的形状，其余分支都有降级。
* **onnx 腿不受影响**：路由只在 `PRIMARY_LEG="crispasr"` 生效，那条腿的语种闸门
  （silero + whisper-tiny `-dl`）原样不动（她 10-14 定的"ort 不换"）。把闸门也换成帧级是后续的事。
* 想退回 10-14 的行为：`ASR_LID_ROUTER=sieve`（每文件只看前 5 s）或 `off`（整批不挑文件）。

### 7.6 判别器的词表里有没有"AED 不支持的语言"（10-16 核）

问：FireRedLID 能不能判出 AED 自己转不了的语言（比如日语）？不能的话判别器就得换。
答：**在词表这一层就能，不用换。**

`model/dict.txt` 逐行核过（本地与校机两份逐字节相同：779 B、`23ef0094…`、120 行、
每行"标签 + 空白或制表 + id"，id 连续 0…119）：

| 分成 | 数量 | 具体 |
|---|---|---|
| 特殊位 | 5 | `<blank>` `<unk>` `<pad>` `<sos>` `<eos>`（id 0-4） |
| 汉语位 | 8 | `zh`=8，方言 token `xinan`=10 `mandarin`=14 `min`=15 `wu`=16 `xiang`=17 `yue`=18 `north`=19 |
| `other` | 1 | id=9，**真实类别**（不属于下面那 106 个），判到它就走兜底 |
| 其他语言码 | 106 | 102 个两字母 + 4 个三字母（`ceb` `haw` `sco` `war`）；**范围外的就在里面**：`ja`=11 `ko`=12 `ru`=13 `th`=83 `vi`=90 |

驱动里 `_AED_IN_RANGE` 那份名单是 **AED 的覆盖面**，不是判别器的输出空间，两者不是一回事；
`lid_onnx.py:240 detok()` 只做 id→标签拼接，`_hyps_to_result()`（同文件 :243-249）连 n-best
一起给，全链没有任何一处按 AED 的覆盖面截过候选。所以"这段是日语"这个判决是**拿得到的**，
路由数的"范围外子段"全都可数 ⇒ 换引擎的判据不需要另找一个语种模型。

顺手一条与筛子不同的：ONNX 这条的 `confidence` 是逐 token 置信度均值（`lid_onnx.py:233-235`），
不是 crispasr 那个写死的 1.0（`src/firered_lid.cpp:90`）—— 就是说这一层**本来还能再加一道
置信度闸门**，本轮没加：她定的判据是"范围外满 3 段"，段数已经是误判的第一道闸，再加一道
没有对应语料可以定阈值。

**还欠的（C）**：词表里有 `ja` ≠ 判得准 `ja`。FireRedLID 在这批小红书素材上对日/韩的召回
一个数都没有；旧记录里那组"日语 2/4"是 whisper-tiny 时代的读数，挂在别的模型上，不能算这件的。
最伤的失手方向也依旧收不住：**外语判成 zh** ⇒ 落进范围内 ⇒ 交给 AED 用汉字编造、退出码 0、
正常落盘。帧级路由改的是"看到多少音频"（整条 vs 前 5 s），不改"判得准不准"，它补上了
"片头是音乐"那一类盲区，没补这一类 —— 满 3 段的门槛只在判出的标签确实落在范围外时才起作用。
要补就一次测量：拿一批已知是日/韩的素材，只跑路由不跑转写，数"范围外段数/总段数"与实际语种的
一致率（校机在批次空闲时段可做，或等笔记本补齐那 3.55 GB 路由权重之后）。

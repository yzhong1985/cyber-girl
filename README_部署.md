# 赛博 AI 女友 · 本地部署（WSL2 + RTX 5090 适配版）

基于 [freedidi 教程](https://www.freedidi.com/24928.html) + 你提供的定制脚本，已针对
**WSL2 / Python 3.12 系统 / RTX 5090 (Blackwell)** 做了适配。

## 流水线
```
你说话 → ① VAD(Silero) → ② STT(Whisper large-v3-turbo) → ③ LLM(Qwen3-14B-Q6_K)
        → ④ TTS(Qwen3-TTS 1.7B, faster-qwen3-tts) → 你听到声音
```

## 相对教程的关键适配
| 项 | 教程 | 本机实际做法 |
|---|---|---|
| Python | 3.11 | 系统是 3.12，用 `uv` 建了独立的 `.venv`(3.11) |
| PyTorch | 普通版 | 5090=Blackwell，装了 **cu128 版 torch 2.11**（否则 GPU 用不了）|
| speech-to-speech | 最新版 | 官方已重构，checkout 教程同期 commit `931e0b5` |
| Qwen3-TTS 推理 | `qwen_tts`(已过时) | 换成官方 `faster-qwen3-tts` 流式 API，重写了 `_synth()` |
| llama.cpp 引擎 | Windows CUDA exe | 无 Linux CUDA 预编译，用 **Windows 原生 exe**，WSL 经网关 IP 调用 |
| 音频 | Windows 直连 | 走 **WSLg PulseAudio**（`PULSE_SERVER=/mnt/wslg/PulseServer`）|

## 目录
```
cyberGirl/
├── .venv/                     Python 3.11 虚拟环境
├── llama-win/                 Windows CUDA llama-server.exe + DLL
├── start_live2d.sh            启动「纯语音 + Live2D 版」
├── start_avatar.sh            启动「真人形象(Ditto)版」
├── stop.sh / status.sh        停止 / 状态
└── speech-to-speech/          底座项目(commit 931e0b5) + 改造
    ├── s2s_pipeline.py        已改 3 处(注入 REGISTRY / qwen3 分支 / serve_panel)
    ├── voice_registry.py      角色热切换 + 面板(补了 current_ref_text)
    ├── characters.json        角色配置(补了 ref_text，LLM 改 qwen3-14b-q6)
    ├── TTS/qwen3_tts_handler.py  重写版(faster-qwen3-tts + registry 热切换)
    ├── voices/                参考音频(当前为占位英文样本)
    └── models/
        ├── qwen3-tts/         Qwen3-TTS-1.7B-Base
        └── llm/Qwen3-14B-Q6_K.gguf
```

## 启动 / 停止（推荐：一键脚本）
```bash
bash start_live2d.sh   # 纯语音 + Live2D 版：LLM 引擎 + 主管线（Ctrl+C 退出对话）
bash start_avatar.sh   # 真人形象(Ditto)版：LLM + Ditto 服务(8902) + avatar 模式管线
bash start_avatar.sh --skip-mask   # 同上但关抠图：用照片原背景，每句快 ~40%(省 2s+)
bash stop.sh           # 一键停止：管线 + Ditto + Windows llama-server，释放显存
bash status.sh         # 查看各组件状态 + 显存占用
```
- 两个启动脚本都会自动拉起 Windows llama-server 并等它就绪，再启动管线；
  首次启动留意 Windows 防火墙弹窗，点「允许访问」。
- `start_live2d.sh`：看到 `切换面板: http://127.0.0.1:8900/` 即成功，对麦克风说话。
  两个网页：`http://127.0.0.1:8900/` 热切换 小满/凛；`http://127.0.0.1:8901/` Live2D 形象（口型跟说话同步）。
- `start_avatar.sh`：浏览器开 `http://127.0.0.1:8902/` 点「进入」，对麦克风说话，她用照片近实时开口回复。
- 依赖前置（一次性，需 sudo）：`sudo apt install -y libportaudio2 libasound2-plugins`

## 配置（改设置只动 config.env）
常调项都集中在项目根目录 **`config.env`**（纯 `KEY=VALUE`，两个启动脚本都会读它）。改完保存、`stop.sh` 再重启对应脚本即生效。主要项：

| 项 | 说明 |
|---|---|
| `PERSONA` | 用哪个角色（`characters.json` 里的 `xiaoman`/`linlin`），性格/回复长度在 characters.json 改 |
| `AVATAR_PHOTO` | 数字人照片路径 |
| `AVATAR_MATTE` | `1`=抠图合成背景(好看/慢)、`0`=照片原背景(快~40%)。命令行 `--skip-mask`/`--matte` 可临时覆盖 |
| `STT_LANGUAGE` | 固定识别语言 `zh`/`en`/…，或 `auto` 自动检测 |
| `VAD_THRESH` / `MIN_SPEECH_MS` / `MIN_SILENCE_MS` | 断句灵敏度 |
| `LLM_MODEL` / `LLM_GGUF` / `LLM_CTX` | 模型别名 / GGUF 路径 / 上下文长度 |

删掉 `config.env` 也能跑（脚本内置同样的默认值）。

## 换成你自己的音色
```bash
# 5-10 秒纯净人声，裁成单声道 24kHz
ffmpeg -i 你的音频.mp3 -ss 0 -t 8 -ac 1 -ar 24000 speech-to-speech/voices/xiaoman.wav
```
然后编辑 `characters.json`，把该角色的 `ref_text` 改成这段音频**实际说的那句话的文字稿**
（faster-qwen3-tts 靠它做声音克隆）。面板点一下即热生效，无需重启。

## Live2D 形象（口型同步）
- 打开 `http://127.0.0.1:8901/` 看形象；管线播放 TTS 时按**音量包络**驱动嘴巴开合（`ParamMouthOpenY`）。
- 组成：`speech-to-speech/live2d/`（`lib/` 运行时 + `model/Hiyori/` 模型 + `index.html`）、
  `live2d_panel.py`（HTTP + SSE 口型流）、`connections/local_audio_streamer.py`（播放时算音量推给页面）。
- **换模型**：把新的 Cubism4 模型放到 `live2d/model/你的模型/`，改 `index.html` 里
  `Live2DModel.from('model/你的模型/xxx.model3.json')`。嘴部参数若不叫 `ParamMouthOpenY`，
  在 `index.html` 的 `setParameterValueById` 处改成模型实际的参数名。
- 口型是"跟响度开合"，非逐音节精确对齐；想更准可后续换成音素级驱动。

## 排错（按流水线四层查）
| 现象 | 原因 | 处理 |
|---|---|---|
| 说话没反应 | VAD 阈值高 | `--thresh` 0.4→0.3 |
| 有文字无回复 | LLM 没通 | 见下方 curl 自检；查 Windows 防火墙 |
| 有回复无声音 | 参考音频/ref_text 问题 | 查 `voices/` 路径与 `characters.json` |
| CUDA out of memory | 显存不够 | LLM 降一档量化 / TTS 换 0.6B |
| PortAudio not found | 缺系统库 | `sudo apt install -y libportaudio2` |

LLM 自检（WSL 里跑）：
```bash
curl http://$(ip route show default | awk '{print $3}'):8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-14b-q6","messages":[{"role":"user","content":"你好"}]}'
```

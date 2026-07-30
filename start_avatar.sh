#!/usr/bin/env bash
# ========================================================================
# 一键启动「真人形象版」: LLM 引擎 + Ditto 形象服务 + 语音管线(avatar模式)
# 说话 → 她用照片近实时开口回复(带身体摆动)。浏览器开 http://127.0.0.1:8902/
#
# 常调设置集中在 config.env(角色/照片/抠图/STT语言/VAD/LLM)。命令行可临时覆盖抠图:
#   bash start_avatar.sh              按 config.env(默认抠图开: 合成到背景, 好看但慢)
#   bash start_avatar.sh --skip-mask  强制关抠图: 照片原背景, 每句快 ~40%(约省 2s+)
#   bash start_avatar.sh --matte      强制开抠图
# ========================================================================
set -e
ROOT=/mnt/e/Projects/cyberGirl

# ---- 命令行参数(优先级高于 config.env): --skip-mask 关抠图 ----
MATTE_CLI=""   # 空=听配置; on/off=命令行强制
for arg in "$@"; do
  case "$arg" in
    --skip-mask|--no-matte|--no-mask) MATTE_CLI="off" ;;
    --mask|--matte)                   MATTE_CLI="on" ;;
    *) echo "未知参数: $arg (可用: --skip-mask / --matte)"; exit 1 ;;
  esac
done

# ---- 读配置(config.env), 缺项用默认值兜底 ----
[ -f "$ROOT/config.env" ] && source "$ROOT/config.env"
: "${PERSONA:=xiaoman}"
: "${AVATAR_PHOTO:=avatar/girl.png}"
: "${AVATAR_MATTE:=1}"
: "${AVATAR_BACKEND:=onnx}"
: "${AVATAR_COMPILE_WARP:=1}"
: "${AVATAR_COMPILE_LMDM:=1}"
: "${STT_MODEL:=openai/whisper-large-v3-turbo}"
: "${STT_LANGUAGE:=zh}"
: "${VAD_THRESH:=0.4}"
: "${MIN_SPEECH_MS:=300}"
: "${MIN_SILENCE_MS:=500}"
: "${LLM_MODEL:=qwen3-14b-q6}"
: "${LLM_GGUF:=E:\\Projects\\cyberGirl\\speech-to-speech\\models\\llm\\Qwen3-14B-Q6_K.gguf}"
: "${LLM_CTX:=8192}"

# 抠图最终决定: 命令行 > config.env
MATTE_FLAG=""
if [ "$MATTE_CLI" = "off" ] || { [ -z "$MATTE_CLI" ] && [ "$AVATAR_MATTE" = "0" ]; }; then
  MATTE_FLAG="--no-matte"
fi

LLM_HOST=$(ip route show default | awk '{print $3}')
# 本机/网关流量绕开系统代理(否则 curl 健康检查走代理回 503 → 误判"已在运行";
# 且 Python urllib/requests 会把 127.0.0.1:8902、LLM 请求也劫持到代理上)
export no_proxy="127.0.0.1,localhost,::1,${LLM_HOST}"
export NO_PROXY="$no_proxy"

# ---- 1) LLM 引擎(Windows CUDA llama-server) ----
if curl -s --max-time 3 "http://${LLM_HOST}:8080/health" 2>/dev/null | grep -q ok; then
  echo "✓ LLM 引擎已在运行"
else
  echo "→ 启动 LLM 引擎(首次留意 Windows 防火墙放行)..."
  nohup "$ROOT/llama-win/llama-server.exe" -m "$LLM_GGUF" \
    --host 0.0.0.0 --port 8080 -ngl 99 -c "$LLM_CTX" -fa on -a "$LLM_MODEL" \
    > "$ROOT/_llm_server.log" 2>&1 &
  echo -n "  等待模型加载 "
  for i in $(seq 1 60); do
    curl -s --max-time 3 "http://${LLM_HOST}:8080/health" 2>/dev/null | grep -q ok && { echo " ✓"; break; }
    echo -n "."; sleep 2
  done
fi

# ---- 2) Ditto 形象服务(独立 py3.10 venv) ----
# 注意: 抠图开/关、生成后端都是服务启动时定死的; 若服务已在运行, 改配置不生效(需先 stop.sh)
if curl -s --max-time 3 http://127.0.0.1:8902/idle.png -o /dev/null 2>/dev/null; then
  echo "✓ Ditto 形象服务已在运行(抠图/后端沿用上次启动; 想改需先 bash stop.sh)"
else
  COMPILE_FLAG=""
  WAIT_HINT="约 30s"
  COMPILE_HINT=""
  if [ "$AVATAR_COMPILE_WARP" = "0" ]; then
    COMPILE_FLAG="$COMPILE_FLAG --no-compile-warp"
  else
    COMPILE_HINT="warp_network"
  fi
  if [ "$AVATAR_COMPILE_LMDM" = "0" ]; then
    COMPILE_FLAG="$COMPILE_FLAG --no-compile-lmdm"
  else
    COMPILE_HINT="${COMPILE_HINT:+$COMPILE_HINT+}lmdm"
  fi
  if [ -n "$COMPILE_HINT" ]; then
    WAIT_HINT="约 100s(含 ${COMPILE_HINT} 编译预热)"
  fi
  if [ -n "$MATTE_FLAG" ]; then
    echo "→ 启动 Ditto 形象服务(抠图关: 照片原背景, 更快; 后端=$AVATAR_BACKEND; 加载模型 $WAIT_HINT)..."
  else
    echo "→ 启动 Ditto 形象服务(抠图开: 合成到背景; 后端=$AVATAR_BACKEND; 加载模型 $WAIT_HINT)..."
  fi
  ( cd "$ROOT/ditto-talkinghead" \
    && export PULSE_SERVER=unix:/mnt/wslg/PulseServer \
    && nohup .venv/bin/python avatar_service.py --avatar "$AVATAR_PHOTO" --port 8902 \
       --backend "$AVATAR_BACKEND" $MATTE_FLAG $COMPILE_FLAG \
       > "$ROOT/_avatar_svc.log" 2>&1 & )
  echo -n "  等待形象服务就绪 "
  for i in $(seq 1 150); do
    curl -s --max-time 3 http://127.0.0.1:8902/idle.png -o /dev/null 2>/dev/null && { echo " ✓"; break; }
    echo -n "."; sleep 2
  done
fi

# ---- 3) 语音管线(avatar 模式: TTS 音频发给 Ditto, 不本地播放) ----
cd "$ROOT/speech-to-speech"
source "$ROOT/.venv/bin/activate"
export PULSE_SERVER=unix:/mnt/wslg/PulseServer
export DITTO_AVATAR_URL="http://127.0.0.1:8902/generate"
PERSONA_PROMPT=$(python -c "import json,sys;print(json.load(open('characters.json'))[sys.argv[1]]['system_prompt'])" "$PERSONA")
echo "→ 启动语音管线(avatar模式, 角色=$PERSONA, STT语言=$STT_LANGUAGE)"
echo "  浏览器打开 http://127.0.0.1:8902/ 点「进入」, 然后对麦克风说话"
exec python s2s_pipeline.py \
  --mode local \
  --stt whisper --stt_model_name "$STT_MODEL" --language "$STT_LANGUAGE" \
  --llm open_api \
  --open_api_base_url "http://${LLM_HOST}:8080/v1" \
  --open_api_api_key local \
  --open_api_model_name "$LLM_MODEL" \
  --open_api_init_chat_role system \
  --open_api_init_chat_prompt "$PERSONA_PROMPT" \
  --tts qwen3 \
  --thresh "$VAD_THRESH" --min_speech_ms "$MIN_SPEECH_MS" --min_silence_ms "$MIN_SILENCE_MS"

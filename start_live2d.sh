#!/usr/bin/env bash
# ========================================================================
# 一键启动「纯语音 + Live2D 版」: LLM 引擎(Windows CUDA llama-server) + 主语音管线
# 网页: 8900 角色切换 / 8901 Live2D 形象。真人形象(Ditto)版见 start_avatar.sh
# 用法：bash start_live2d.sh  (Ctrl+C 退出主管线；LLM 引擎后台续跑，用 stop.sh 停)
# ========================================================================
set -e
ROOT=/mnt/e/Projects/cyberGirl

# ---- 读配置(config.env), 缺项用默认值兜底 ----
[ -f "$ROOT/config.env" ] && source "$ROOT/config.env"
: "${PERSONA:=xiaoman}"
: "${STT_MODEL:=openai/whisper-large-v3-turbo}"
: "${STT_LANGUAGE:=zh}"
: "${VAD_THRESH:=0.4}"
: "${MIN_SPEECH_MS:=300}"
: "${MIN_SILENCE_MS:=500}"
: "${LLM_MODEL:=qwen3-14b-q6}"
: "${LLM_GGUF:=E:\\Projects\\cyberGirl\\speech-to-speech\\models\\llm\\Qwen3-14B-Q6_K.gguf}"
: "${LLM_CTX:=8192}"

cd "$ROOT/speech-to-speech"
source "$ROOT/.venv/bin/activate"
export PULSE_SERVER=unix:/mnt/wslg/PulseServer
LLM_HOST=$(ip route show default | awk '{print $3}')
# 本机/网关流量绕开系统代理(否则 curl 健康检查走代理回 503 误判，Python 请求也被劫持)
export no_proxy="127.0.0.1,localhost,::1,${LLM_HOST}"
export NO_PROXY="$no_proxy"

# ---- 1) LLM 引擎：没在跑就启动 ----
if curl -s --max-time 3 "http://${LLM_HOST}:8080/health" 2>/dev/null | grep -q ok; then
  echo "✓ LLM 引擎已在运行 (${LLM_HOST}:8080)"
else
  echo "→ 启动 LLM 引擎 (Windows llama-server, RTX 5090)..."
  echo "  ⚠️ 首次启动留意 Windows 防火墙弹窗，点「允许访问」"
  nohup "$ROOT/llama-win/llama-server.exe" -m "$LLM_GGUF" \
    --host 0.0.0.0 --port 8080 -ngl 99 -c "$LLM_CTX" -fa on -a "$LLM_MODEL" \
    > "$ROOT/_llm_server.log" 2>&1 &
  echo -n "  等待模型加载 "
  for i in $(seq 1 60); do
    if curl -s --max-time 3 "http://${LLM_HOST}:8080/health" 2>/dev/null | grep -q ok; then echo " ✓"; break; fi
    echo -n "."; sleep 2
  done
fi

# ---- 2) 主管线：前台运行 ----
PERSONA_PROMPT=$(python -c "import json,sys;print(json.load(open('characters.json'))[sys.argv[1]]['system_prompt'])" "$PERSONA")
echo "→ 启动主管线(角色=$PERSONA, STT语言=$STT_LANGUAGE)... 看到「切换面板: http://127.0.0.1:8900/」后对麦克风说话"
echo "  (Ctrl+C 结束对话；面板可热切换 小满/凛)"
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

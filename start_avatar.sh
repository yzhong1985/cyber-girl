#!/usr/bin/env bash
# ========================================================================
# 一键启动「真人形象版」: LLM 引擎 + Ditto 形象服务 + 语音管线(avatar模式)
# 说话 → 她用照片近实时开口回复(带身体摆动)。浏览器开 http://127.0.0.1:8902/
#
# 常调设置集中在 config.env(角色/照片/抠图/STT语言/VAD/LLM)。命令行可临时覆盖抠图:
#   bash start_avatar.sh              按 config.env(默认抠图开: 合成到背景, 好看但慢)
#   bash start_avatar.sh --skip-mask  强制关抠图: 照片原背景, 每句快 ~40%(约省 2s+)
#   bash start_avatar.sh --matte      强制开抠图
#   bash start_avatar.sh --web        网页版: 浏览器采麦(WebSocket), 自签HTTPS供局域网访问
#   bash start_avatar.sh --engine=livetalking   命令行临时切形象后端, 不改配置文件
#
# 形象后端(config.env 的 AVATAR_ENGINE=ditto/livetalking)可以换成 LiveTalking
# (独立项目 /mnt/e/Projects/livetalking-lab, wav2lip256/musetalk, 延迟低画质糊)
# 代替 Ditto(扩散模型, 延迟高画质自然)。engine=livetalking 时画面在它自己的页面
# (127.0.0.1:8010)看, 不是 127.0.0.1:8902; 暂不支持 --web(见下面代码里的报错提示)。
# ========================================================================
set -e
ROOT=/mnt/e/Projects/cyberGirl
SCRIPT_START=$(date +%s)
log() { echo "[$(date +%H:%M:%S)] $*"; }

# 外部传入的 DITTO_AVATAR_URL(比如指向 livetalking-lab 的桥接服务)优先于本脚本
# 自己的 Ditto 形象服务: 设了就跳过第2步(省~100s编译预热+显存), 第3步直接用这个值,
# 不会像以前那样被脚本自己的 export 悄悄覆盖回 Ditto 的地址。
EXTERNAL_AVATAR_URL="${DITTO_AVATAR_URL:-}"

# ---- 命令行参数(优先级高于 config.env): --skip-mask 关抠图, --web 开网页版, --engine=切后端 ----
MATTE_CLI=""     # 空=听配置; on/off=命令行强制
WEB_CLI=""       # 空=听配置; on=命令行强制开网页版
ENGINE_CLI=""    # 空=听配置; ditto/livetalking=命令行强制
for arg in "$@"; do
  case "$arg" in
    --skip-mask|--no-matte|--no-mask) MATTE_CLI="off" ;;
    --mask|--matte)                   MATTE_CLI="on" ;;
    --web)                            WEB_CLI="on" ;;
    --engine=*)                       ENGINE_CLI="${arg#--engine=}" ;;
    *) echo "未知参数: $arg (可用: --skip-mask / --matte / --web / --engine=ditto|livetalking)"; exit 1 ;;
  esac
done

# ---- 读配置(config.env), 缺项用默认值兜底 ----
[ -f "$ROOT/config.env" ] && source "$ROOT/config.env"
: "${PERSONA:=xiaoman}"
: "${AVATAR_ENGINE:=ditto}"
[ -n "$ENGINE_CLI" ] && AVATAR_ENGINE="$ENGINE_CLI"
: "${LIVETALKING_MODEL:=wav2lip}"
: "${LIVETALKING_ROOT:=/mnt/e/Projects/livetalking-lab}"
: "${AVATAR_PHOTO:=avatar/girl.png}"
: "${AVATAR_MATTE:=1}"
: "${AVATAR_BACKEND:=onnx}"
: "${AVATAR_COMPILE_WARP:=1}"
: "${AVATAR_COMPILE_LMDM:=1}"
: "${AVATAR_FILLER:=1}"
: "${AVATAR_FILLER_DELAY_MIN:=1.6}"
: "${AVATAR_FILLER_DELAY_MAX:=2.6}"
: "${AVATAR_STREAM_SENTENCES:=0}"
: "${STT_MODEL:=openai/whisper-large-v3-turbo}"
: "${STT_LANGUAGE:=zh}"
: "${VAD_THRESH:=0.4}"
: "${MIN_SPEECH_MS:=300}"
: "${MIN_SILENCE_MS:=500}"
: "${LLM_MODEL:=qwen3-14b-q6}"
: "${LLM_GGUF:=E:\\Projects\\cyberGirl\\speech-to-speech\\models\\llm\\Qwen3-14B-Q6_K.gguf}"
: "${LLM_CTX:=8192}"
: "${WEB_MODE:=0}"
: "${WS_PORT:=8765}"

# 网页版最终决定: 命令行 > config.env
WEB=0
if [ "$WEB_CLI" = "on" ] || { [ -z "$WEB_CLI" ] && [ "$WEB_MODE" = "1" ]; }; then
  WEB=1
fi

# --web + AVATAR_ENGINE=livetalking 这个组合暂不支持: --web 模式下麦克风采集靠
# player.html 的 WebSocket 页面, 而这个页面是 Ditto 的 avatar_service.py 顺带serve
# 出来的——engine=livetalking 时第2步根本不启动 avatar_service.py, 没人serve这个
# 页面, 会静默失败。等 player.html 接入 LiveTalking WebRTC(livetalking-lab项目
# docs/进度.md 里的路线图)才能支持。engine=livetalking 现在只支持本机麦克风模式。
if [ "$WEB" = "1" ] && [ "$AVATAR_ENGINE" = "livetalking" ]; then
  echo "✗ 暂不支持 --web + AVATAR_ENGINE=livetalking 组合(麦克风采集页面没人serve)。"
  echo "  engine=livetalking 请去掉 --web / WEB_MODE=0, 用本机麦克风模式。"
  exit 1
fi

# ---- 网页版: 首次用自签证书(证书不存在才生成; 换了局域网/IP变了要手动删掉 certs/ 重生成) ----
TLS_CERT=""; TLS_KEY=""; TLS_FLAG_AVATAR=""; TLS_FLAG_WS=""; LAN_IPS=""
if [ "$WEB" = "1" ]; then
  mkdir -p "$ROOT/certs"
  TLS_CERT="$ROOT/certs/dev.crt"; TLS_KEY="$ROOT/certs/dev.key"
  # 尽力探测 Windows 侧的局域网 IP(局域网设备连的是 Windows, 不是 WSL 内部 IP); 探测失败不影响启动
  LAN_IPS=$(/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
    "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object {\$_.InterfaceAlias -notmatch 'Loopback|vEthernet|WSL' -and \$_.IPAddress -notmatch '^169\.254\.'}).IPAddress" \
    2>/dev/null | tr -d '\r') || true
  if [ ! -f "$TLS_CERT" ] || [ ! -f "$TLS_KEY" ]; then
    echo "→ 首次生成自签 HTTPS 证书(证书不含私密信息, 只是让浏览器认为连接加密)..."
    SAN="DNS:localhost,IP:127.0.0.1"
    for ip in $LAN_IPS; do SAN="$SAN,IP:$ip"; done
    openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
      -keyout "$TLS_KEY" -out "$TLS_CERT" \
      -subj "/CN=cybergirl.local" -addext "subjectAltName=$SAN" >/dev/null 2>&1
    echo "  ✓ 证书已生成(覆盖: $SAN)"
    echo "  (若之后局域网 IP 变了, 证书不会自动更新: 删掉 $ROOT/certs/ 重新运行本脚本即可重新生成)"
  fi
  TLS_FLAG_AVATAR="--tls-cert $TLS_CERT --tls-key $TLS_KEY"
  TLS_FLAG_WS="--ws_tls_cert $TLS_CERT --ws_tls_key $TLS_KEY"
fi

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
  log "✓ LLM 引擎已在运行"
else
  log "→ 启动 LLM 引擎(首次留意 Windows 防火墙放行)..."
  nohup "$ROOT/llama-win/llama-server.exe" -m "$LLM_GGUF" \
    --host 0.0.0.0 --port 8080 -ngl 99 -c "$LLM_CTX" -fa on -a "$LLM_MODEL" \
    > "$ROOT/_llm_server.log" 2>&1 &
  echo -n "  等待模型加载 "
  for i in $(seq 1 60); do
    curl -s --max-time 3 "http://${LLM_HOST}:8080/health" 2>/dev/null | grep -q ok && { echo " ✓ [$(date +%H:%M:%S)]"; break; }
    echo -n "."; sleep 2
  done
fi

# ---- 2) 形象服务: Ditto(独立 py3.10 venv) 或 LiveTalking(独立项目 livetalking-lab) ----
# 注意: 抠图开/关、生成后端都是服务启动时定死的; 若服务已在运行, 改配置不生效(需先 stop.sh)
SCHEME="http"; CURL_INSECURE=""
[ "$WEB" = "1" ] && { SCHEME="https"; CURL_INSECURE="-k"; }
if [ -n "$EXTERNAL_AVATAR_URL" ]; then
  log "→ 跳过形象服务启动(DITTO_AVATAR_URL 已外部指定: $EXTERNAL_AVATAR_URL)"
elif [ "$AVATAR_ENGINE" = "livetalking" ]; then
  # ---- 2a) LiveTalking: 直接复用 livetalking-lab 自己的 start.sh, 不重复实现
  #          健康检查/等待逻辑(它自己就会判断"已在运行"跳过重启、打印进度点)。----
  if [ ! -d "$LIVETALKING_ROOT" ]; then
    echo "✗ 找不到 LIVETALKING_ROOT=$LIVETALKING_ROOT (config.env 配置的路径), 检查一下"
    exit 1
  fi
  log "→ 启动 LiveTalking(livetalking-lab, model=$LIVETALKING_MODEL)..."
  ( cd "$LIVETALKING_ROOT" && bash start.sh --model "$LIVETALKING_MODEL" )
  # ---- 2a续) 轻量版 avatar_service.py: 光是 LiveTalking 本身不 serve player.html,
  #            这里补一个只管页面、不加载 Ditto 模型的轻量实例, 保住 127.0.0.1:8902
  #            这个书签不变(player.html 加载时探测 /backend_info 决定画面走哪条路,
  #            见 Phase2 的 player.html 改动)。----
  if curl -s $CURL_INSECURE --max-time 3 "${SCHEME}://127.0.0.1:8902/backend_info" -o /dev/null 2>/dev/null; then
    log "✓ 播放页服务(lite)已在运行"
  else
    log "→ 启动播放页服务(lite, 只serve页面, 不加载Ditto模型)..."
    ( cd "$ROOT/ditto-talkinghead" \
      && nohup .venv/bin/python avatar_service.py --lite --port 8902 \
         --livetalking-url "http://127.0.0.1:8010" $TLS_FLAG_AVATAR \
         > "$ROOT/_avatar_svc.log" 2>&1 & )
    echo -n "  等待播放页服务就绪 "
    for i in $(seq 1 30); do
      curl -s $CURL_INSECURE --max-time 3 "${SCHEME}://127.0.0.1:8902/backend_info" -o /dev/null 2>/dev/null && { echo " ✓ [$(date +%H:%M:%S)]"; break; }
      echo -n "."; sleep 1
    done
  fi
elif curl -s $CURL_INSECURE --max-time 3 "${SCHEME}://127.0.0.1:8902/idle.png" -o /dev/null 2>/dev/null; then
  log "✓ Ditto 形象服务已在运行(抠图/后端/网页版沿用上次启动; 想改需先 bash stop.sh)"
else
  # ---- 2b) Ditto ----
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
    log "→ 启动 Ditto 形象服务(抠图关: 照片原背景, 更快; 后端=$AVATAR_BACKEND; 加载模型 $WAIT_HINT)..."
  else
    log "→ 启动 Ditto 形象服务(抠图开: 合成到背景; 后端=$AVATAR_BACKEND; 加载模型 $WAIT_HINT)..."
  fi
  ( cd "$ROOT/ditto-talkinghead" \
    && export PULSE_SERVER=unix:/mnt/wslg/PulseServer \
    && nohup .venv/bin/python avatar_service.py --avatar "$AVATAR_PHOTO" --port 8902 \
       --backend "$AVATAR_BACKEND" $MATTE_FLAG $COMPILE_FLAG $TLS_FLAG_AVATAR \
       > "$ROOT/_avatar_svc.log" 2>&1 & )
  echo -n "  等待形象服务就绪 "
  for i in $(seq 1 150); do
    curl -s $CURL_INSECURE --max-time 3 "${SCHEME}://127.0.0.1:8902/idle.png" -o /dev/null 2>/dev/null && { echo " ✓ [$(date +%H:%M:%S)]"; break; }
    echo -n "."; sleep 2
  done
fi

# ---- 3) 语音管线 ----
# 本机模式: --mode local(用本机麦克风/WSLg音频); 网页版: --mode websocket(浏览器采麦, 端口8765)
# 两种模式下 TTS 音频都直接 POST 给 Ditto, 不走管线自己的音频输出
cd "$ROOT/speech-to-speech"
source "$ROOT/.venv/bin/activate"
export PULSE_SERVER=unix:/mnt/wslg/PulseServer
# 形象后端地址: 外部覆盖 > AVATAR_ENGINE 对应的默认地址
if [ -n "$EXTERNAL_AVATAR_URL" ]; then
  AVATAR_URL="$EXTERNAL_AVATAR_URL"
elif [ "$AVATAR_ENGINE" = "livetalking" ]; then
  AVATAR_URL="http://127.0.0.1:9000/generate"   # livetalking-lab 桥接服务默认端口; 改了要同步这里
else
  AVATAR_URL="${SCHEME}://127.0.0.1:8902/generate"
fi
export DITTO_AVATAR_URL="$AVATAR_URL"
export AVATAR_FILLER="$AVATAR_FILLER"
export AVATAR_FILLER_DELAY_MIN="$AVATAR_FILLER_DELAY_MIN"
export AVATAR_FILLER_DELAY_MAX="$AVATAR_FILLER_DELAY_MAX"
export AVATAR_STREAM_SENTENCES="$AVATAR_STREAM_SENTENCES"
PERSONA_PROMPT=$(python -c "import json,sys;print(json.load(open('characters.json'))[sys.argv[1]]['system_prompt'])" "$PERSONA")
MODE_FLAG="--mode local"
if [ "$WEB" = "1" ]; then
  MODE_FLAG="--mode websocket --ws_host 0.0.0.0 --ws_port $WS_PORT $TLS_FLAG_WS"
fi
log "→ 启动语音管线(角色=$PERSONA, STT语言=$STT_LANGUAGE, 网页版=$([ "$WEB" = "1" ] && echo 开 || echo 关)); LLM+Ditto已耗时 $(( $(date +%s) - SCRIPT_START ))s, 接下来还要加载STT/LLM客户端/TTS模型(下面Python自己的日志带时间戳, 通常再需30~60s)"
if [ "$WEB" = "1" ]; then
  LAN_IP=$(echo "$LAN_IPS" | head -1)
  echo "  本机浏览器打开 https://127.0.0.1:8902/ 点「进入」(会弹证书不受信任警告, 选择继续前往)"
  if [ -n "$LAN_IP" ]; then
    echo "  局域网设备(还需下面两步, 只用做一次):"
    echo "   1. Windows 管理员 PowerShell 执行端口转发+防火墙放行(WSL IP 重启会变, 每次重启WSL都要重新执行):"
    WSL_IP=$(hostname -I | awk '{print $1}')
    echo "      netsh interface portproxy add v4tov4 listenport=8902 listenaddress=0.0.0.0 connectport=8902 connectaddress=$WSL_IP"
    echo "      netsh interface portproxy add v4tov4 listenport=$WS_PORT listenaddress=0.0.0.0 connectport=$WS_PORT connectaddress=$WSL_IP"
    echo "      netsh advfirewall firewall add rule name=cybergirl-web dir=in action=allow protocol=TCP localport=8902,$WS_PORT"
    echo "   2. 其它设备浏览器先打开 https://${LAN_IP}:${WS_PORT}/ 点「继续前往」信任证书(WS连接本身无法弹这个提示, 必须先单独访问一次)"
    echo "   3. 再打开 https://${LAN_IP}:8902/ 点「进入」, 同样先信任证书, 然后允许麦克风权限"
  fi
elif [ -n "$EXTERNAL_AVATAR_URL" ]; then
  echo "  形象后端是外部指定的($EXTERNAL_AVATAR_URL), 画面不在 127.0.0.1:8902, 去对应服务自己的页面看; 麦克风照常本机采集"
elif [ "$AVATAR_ENGINE" = "livetalking" ]; then
  echo "  浏览器打开 http://127.0.0.1:8902/ 点「进入」, 然后对麦克风说话(画面走 LiveTalking, 页面跟 Ditto 版是同一个)"
else
  echo "  浏览器打开 http://127.0.0.1:8902/ 点「进入」, 然后对麦克风说话"
fi
exec python s2s_pipeline.py \
  $MODE_FLAG \
  --stt whisper --stt_model_name "$STT_MODEL" --language "$STT_LANGUAGE" \
  --llm open_api \
  --open_api_base_url "http://${LLM_HOST}:8080/v1" \
  --open_api_api_key local \
  --open_api_model_name "$LLM_MODEL" \
  --open_api_init_chat_role system \
  --open_api_init_chat_prompt "$PERSONA_PROMPT" \
  --tts qwen3 \
  --thresh "$VAD_THRESH" --min_speech_ms "$MIN_SPEECH_MS" --min_silence_ms "$MIN_SILENCE_MS"

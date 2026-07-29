#!/usr/bin/env bash
# 查看 cyber girl 各组件运行状态
LLM_HOST=$(ip route show default | awk '{print $3}')
# 本机/网关流量绕开系统代理(否则 curl 走代理回 503，明明在跑却误报"未运行")
export no_proxy="127.0.0.1,localhost,::1,${LLM_HOST}"
export NO_PROXY="$no_proxy"
echo "== cyber girl 状态 =="

if pgrep -f 's2s_pipeline[.]py' >/dev/null; then
  echo "  主管线:   运行中 ✓ (PID $(pgrep -f 's2s_pipeline[.]py' | head -1))"
else
  echo "  主管线:   未运行"
fi

if curl -s --max-time 3 "http://${LLM_HOST}:8080/health" 2>/dev/null | grep -q ok; then
  echo "  LLM引擎:  运行中 ✓ (http://${LLM_HOST}:8080)"
else
  echo "  LLM引擎:  未运行"
fi

if ss -tln 2>/dev/null | grep -q :8900; then
  echo "  切换面板: http://127.0.0.1:8900/ ✓"
else
  echo "  切换面板: 未运行"
fi

if ss -tln 2>/dev/null | grep -q :8902; then
  echo "  Ditto形象: http://127.0.0.1:8902/ ✓"
else
  echo "  Ditto形象: 未运行"
fi

echo "  GPU:      $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null)"

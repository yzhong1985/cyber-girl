#!/usr/bin/env bash
# ========================================================================
# 一键停止：主语音管线 + LLM 引擎(Windows llama-server)
# 用法：bash stop.sh
# ========================================================================
echo "== 停止 cyber girl =="

# 1) 主管线(WSL python)。用正则转义模式，不会误杀本脚本自身
if pkill -9 -f 's2s_pipeline[.]py' 2>/dev/null; then
  echo "  ✓ 主管线已停"
else
  echo "  · 主管线未在运行"
fi

# 2) Ditto 形象服务(WSL python)
if pkill -9 -f 'avatar_service[.]py' 2>/dev/null; then
  echo "  ✓ Ditto 形象服务已停"
else
  echo "  · Ditto 形象服务未在运行"
fi

# 3) LLM 引擎(Windows 进程)——经互操作 taskkill
if /mnt/c/Windows/System32/taskkill.exe /F /IM llama-server.exe >/dev/null 2>&1; then
  echo "  ✓ LLM 引擎(llama-server.exe)已停"
else
  echo "  · LLM 引擎未在运行"
fi

echo "完成"

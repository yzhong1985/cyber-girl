"""
一次性验证脚本: 用真实的 Qwen3TTSHandler(真实音色克隆) 生成"凛"这个角色的
一句台词, 通过 DITTO_AVATAR_URL 指向的桥接服务送进 LiveTalking, 不走 VAD/STT,
只验证 LLM回复文本 -> TTS -> 桥接服务 -> LiveTalking 这一段真实链路。

用法(要先在 livetalking-lab 那边起好 LiveTalking + bridge/server.py, 并且已经
有一个活跃的 WebRTC session, 见 tests/hold_session.py):

    DITTO_AVATAR_URL=http://127.0.0.1:9000/generate AVATAR_FILLER=0 \
        .venv/bin/python speech-to-speech/_livetalking_bridge_test.py
"""
import logging
import os
import queue
import sys
from threading import Event

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from TTS.qwen3_tts_handler import Qwen3TTSHandler  # noqa: E402

handler = Qwen3TTSHandler(
    stop_event=Event(),
    queue_in=queue.Queue(),
    queue_out=queue.Queue(),
    setup_kwargs=dict(
        should_listen=Event(),
        model_name="/home/yzhong/cybergirl/models/qwen3-tts",
        ref_audio="voices/linlin.wav",
        ref_text="其实我一开始就是定妆的时候把刘海一剪然后两个小边一梳上我一下子就觉得我说我好像回到了初中高中的那个样子",
        language="Chinese",
        device="cuda",
    ),
)

text = "回来了啊。等你半天了，也不说一声。"
print(f"\n[test] 合成并发送: {text!r}\n")
list(handler.process(text))
print("\n[test] 完成，去看 LiveTalking 服务端日志确认渲染了没\n")

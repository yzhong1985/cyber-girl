"""
LLM Output Processor

Intercepts LLM output to:
1. Extract tool calls and send them via text_output_queue
2. Forward clean text to TTS pipeline
3. (avatar 模式) 触发一段跟回复长短匹配的过渡语占位播放
"""

import logging
import os
import random
import ssl
import time
import urllib.request
from threading import Thread

from baseHandler import BaseHandler

logger = logging.getLogger(__name__)


class LMOutputProcessor(BaseHandler):
    """
    Processes LLM output to extract tool calls and forward clean text to TTS.

    Input: (text, language_code, tools) tuples from LLM
    Output: (text, language_code) tuples to TTS
    Side effect: Sends {"type": "assistant_text", "text": ..., "tools": ...} to text_output_queue
    """

    def setup(self, text_output_queue):
        """
        Initialize the processor.

        Args:
            text_output_queue: Queue to send text messages and tool calls
        """
        self.text_output_queue = text_output_queue
        # 铺垫语要放在这里触发(而不是VAD说完话那一刻): 这时候 LLM 已经把真实回复
        # 文字给出来了, 能按回复字数挑一段时长接近的过渡语(短回复配短过渡语,
        # 长回复配长过渡语), 比纯随机自然。LLM 稳态很快(有prompt前缀缓存,
        # 通常<1s), 加上下面的延迟, 用户感知不到"是等LLM才触发"这个先后关系。
        avatar_url = os.environ.get("DITTO_AVATAR_URL")
        self.filler_url = (avatar_url.rsplit("/generate", 1)[0] + "/play_filler") if avatar_url else None
        self.filler_enabled = os.environ.get("AVATAR_FILLER", "1").lower() not in ("0", "false", "no", "off")
        # 触发即播延迟基本为0会显得像"你话音刚落她就抢答", 很假; 模拟真人反应
        # 先"愣"一下再接过渡语。范围随机避免每次都一模一样的节奏。
        self.filler_delay_min = float(os.environ.get("AVATAR_FILLER_DELAY_MIN", "1.6"))
        self.filler_delay_max = float(os.environ.get("AVATAR_FILLER_DELAY_MAX", "2.6"))

    def _trigger_filler(self, text_chunk: str):
        """fire-and-forget 叫 Ditto 按 text_chunk 的字数挑一段时长匹配的过渡语
        推去占位播放; 跟正常的 TTS→Ditto(其它线程)是真并行, 不阻塞这里。"""
        if not (self.filler_url and self.filler_enabled):
            return

        delay = random.uniform(self.filler_delay_min, self.filler_delay_max)
        chars = len(text_chunk or "")
        url = f"{self.filler_url}?chars={chars}"

        def _fire():
            time.sleep(delay)
            try:
                ctx = None
                if url.startswith("https://"):
                    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                urllib.request.urlopen(url, timeout=5, context=ctx).read()
            except Exception as e:
                logger.debug(f"[LMProcessor] 过渡语触发失败(不致命): {e}")

        Thread(target=_fire, daemon=True).start()

    def process(self, lm_output):
        """
        Process LLM output: send text/tools to WebSocket, forward clean text to TTS.

        Args:
            lm_output: Tuple of (text, language_code, tools)

        Yields:
            Tuple of (text, language_code) for TTS
        """
        text_chunk, language_code, tools = lm_output
        logger.debug(f"LM processor: text='{text_chunk}', tools={tools}")

        # Send text + tools to WebSocket clients
        if tools:
            message = {
                "type": "assistant_text",
                "text": text_chunk,
                "tools": tools
            }
            logger.info(f"Sending to clients: text='{text_chunk}', tools={[t['name'] for t in tools]}")
            self.text_output_queue.put(message)
        else:
            message = {
                "type": "assistant_text",
                "text": text_chunk
            }
            logger.debug(f"Sending to clients: text='{text_chunk}' (no tools)")
            self.text_output_queue.put(message)

        if text_chunk and text_chunk.strip():
            self._trigger_filler(text_chunk)

        # Forward clean text to TTS (yield to maintain streaming)
        logger.debug(f"Forwarding to TTS: '{text_chunk}'")
        yield (text_chunk, language_code)

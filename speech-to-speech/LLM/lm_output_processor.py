"""
LLM Output Processor

Intercepts LLM output to:
1. Extract tool calls and send them via text_output_queue
2. Forward clean text to TTS pipeline(整段, 或按 AVATAR_STREAM_SENTENCES 开关切句分段)
3. (avatar 模式) 触发一段跟回复长短匹配的过渡语占位播放
"""

import logging
import os
import random
import re
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
    Output: (text, language_code, is_last) tuples to TTS —— 开关关闭时永远只有一个
        元素(is_last=True), 跟原来行为等价; 开关打开时按句切开陆续 yield, 最后
        一个 is_last=True。
    Side effect: Sends {"type": "assistant_text", "text": ..., "tools": ...} to text_output_queue
    """

    _SENT_RE = re.compile(r"[^。！？!?]*[。！？!?]+")

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
        # 分句流式播放开关: 关(默认)=整段回复攒够了才有画面(原来的路线,
        # 稳定但~14s空白); 开=按句切开, 第一句生成完就先播, 后面的句子边播
        # 边生成(见 qwen3_tts_handler.py 的 is_last 处理)。新功能, 默认关,
        # 出问题改回0就能秒回退到原路线。
        self.stream_sentences = os.environ.get("AVATAR_STREAM_SENTENCES", "0").lower() not in ("0", "false", "no", "off")

    def _split_sentences(self, text: str, min_len: int = 6):
        """按中文标点(。！？!?)切句; 太短的碎句(比如单独一个"嗯。")并进下一句,
        避免切太碎——每句都要付一次 Ditto 的固定开销(人脸配准/编解码), 切太碎
        总耗时反而可能比不切还长。"""
        text = (text or "").strip()
        if not text:
            return []
        parts = self._SENT_RE.findall(text)
        consumed_len = sum(len(p) for p in parts)
        remainder = text[consumed_len:].strip()
        if remainder:
            parts.append(remainder)
        parts = [p.strip() for p in parts if p.strip()]

        merged, buf = [], ""
        for p in parts:
            buf = (buf + p) if buf else p
            if len(buf) >= min_len:
                merged.append(buf)
                buf = ""
        if buf:
            if merged:
                merged[-1] += buf
            else:
                merged.append(buf)
        return merged

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
            (text, language_code, is_last) 三元组给 TTS; is_last 标记这是不是
            这轮回复的最后一段(只有最后一段才该让 TTS 播完后放开麦克风)。
        """
        text_chunk, language_code, tools = lm_output
        logger.debug(f"LM processor: text='{text_chunk}', tools={tools}")

        # Send text + tools to WebSocket clients(字幕/文本消息还是发整段, 不分句)
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

        if not text_chunk or not text_chunk.strip():
            return

        if self.stream_sentences:
            sentences = self._split_sentences(text_chunk) or [text_chunk]
        else:
            sentences = [text_chunk]

        self._trigger_filler(sentences[0])

        # Forward clean text to TTS (yield to maintain streaming)
        n = len(sentences)
        for i, sentence in enumerate(sentences):
            is_last = (i == n - 1)
            logger.debug(f"Forwarding to TTS: '{sentence}' (is_last={is_last})")
            yield (sentence, language_code, is_last)

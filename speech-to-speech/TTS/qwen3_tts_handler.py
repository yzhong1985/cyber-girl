"""
TTS/qwen3_tts_handler.py

把 Qwen3-TTS 接进 huggingface/speech-to-speech 的 TTS handler。
CUDA 上走官方低延迟推理库 faster-qwen3-tts（流式声音克隆）。

依赖：
    pip install faster-qwen3-tts soundfile scipy

⚠️ _synth() 里的推理调用是唯一需要跟官方仓库对齐的地方。
   已按 faster-qwen3-tts>=0.2.6 的 generate_voice_clone_streaming API 对齐，
   并对可能变动的 kwargs 做了容错回退。
"""

import logging
import math
import os
import re
import ssl
import time
import unicodedata
import urllib.request
from threading import Event, Thread

import numpy as np
import torch
from scipy.signal import resample_poly

from baseHandler import BaseHandler

logger = logging.getLogger(__name__)

# 下游 socket / 本地播放发送端写死 16kHz int16，这里必须重采样过去
TARGET_SR = 16000


class Qwen3TTSHandler(BaseHandler):
    """
    输入：LLM 吐出的一句话（str，或 (str, lang_code) 元组）
    输出：int16 PCM 分块，16kHz 单声道
    """

    def setup(
        self,
        should_listen: Event,
        model_name: str = "./models/qwen3-tts",
        device: str = "cuda",
        torch_dtype: str = "float16",
        ref_audio: str = "voices/xiaoman.wav",
        ref_text: str = "",
        language: str = "Chinese",
        blocksize: int = 512,
        streaming_chunk_size: int = 8,
        max_new_tokens: int = 1536,
        attn_implementation: str = "sdpa",
        registry=None,          # 传入 VoiceRegistry 就能热切换音色
    ):
        self.should_listen = should_listen
        self.device = device
        self.blocksize = blocksize
        self.language = language
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.streaming_chunk_size = streaming_chunk_size
        self.max_new_tokens = max_new_tokens
        self.registry = registry

        # avatar 模式: 设了 DITTO_AVATAR_URL 就把音频发给 Ditto 形象服务, 不本地播放
        self.avatar_url = os.environ.get("DITTO_AVATAR_URL")
        self.avatar_mode = bool(self.avatar_url)
        self.avatar_base = self.avatar_url.rsplit("/generate", 1)[0] if self.avatar_mode else None
        # 铺垫语("让我想想哈"之类): 说完话到真正回复中间那段空白(~14s)太安静,
        # 预生成好一小池过渡语视频, VAD一检测到说完话就立刻推一段占位, 真正的
        # STT→LLM→TTS→Ditto 在后台并行跑。默认开, AVATAR_FILLER=0 可关。
        self.filler_enabled = os.environ.get("AVATAR_FILLER", "1").lower() not in ("0", "false", "no", "off")
        if self.avatar_mode:
            logger.info(f"[TTS] avatar 模式已开: 音频→ {self.avatar_url}")

        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

        from faster_qwen3_tts import FasterQwen3TTS

        # 不同版本 from_pretrained 的 kwargs 略有出入，做一层容错
        try:
            self.model = FasterQwen3TTS.from_pretrained(
                model_name,
                device=device,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )
        except TypeError:
            self.model = FasterQwen3TTS.from_pretrained(model_name, device=device, dtype=dtype)

        logger.info(f"Qwen3-TTS loaded (faster-qwen3-tts): {model_name} on {device}")
        self.warmup()

        if self.avatar_mode and self.filler_enabled:
            self._pregenerate_fillers()
            if self.registry is not None:
                # 切角色 = 换音色, 池子里旧音色的过渡语得重新合成; 放后台线程避免
                # 卡住切换面板那个HTTP请求(重新生成要跑好几秒Ditto推理)。
                self.registry.register_callback(
                    lambda _name: Thread(target=self._pregenerate_fillers, daemon=True).start()
                )

    def warmup(self):
        """先跑一发，把 CUDA kernel 和图编译的开销吃掉。
        不做这一步，第一句话会卡 2-3 秒，录视频很难看。"""
        logger.info("Warming up Qwen3TTSHandler")
        try:
            self._synth("你好。")
            if self.device == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Warmup failed (不致命，继续): {e}")

    # ---------- 热切换入口 ----------

    def _current_ref(self):
        """registry 里的当前音色优先于启动参数"""
        if self.registry is not None:
            return self.registry.current_ref_audio()
        return self.ref_audio

    def _current_ref_text(self):
        if self.registry is not None:
            return self.registry.current_ref_text()
        return self.ref_text

    _DEFAULT_FILLERS = ["嗯，让我想想", "让我想想哈", "等一下"]

    def _current_fillers(self):
        if self.registry is not None:
            return self.registry.current_fillers()
        return list(self._DEFAULT_FILLERS)

    # 官方 _estimate_max_new_tokens 的移植：按文本长度动态估算生成上限，
    # 避免模型不收口时一路合成到固定上限（会得到一两分钟的废音频）。
    _MIN_UTTERANCE_TOKENS = 240        # 下限(占位音色不停时的最坏时长≈19s)
    _TOKENS_PER_SECOND = 12.5
    _CHARS_PER_SECOND = 14.0
    _WORDS_PER_SECOND = 2.6
    _SAFETY = 1.35

    def _estimate_max_new_tokens(self, text: str) -> int:
        text = (text or "").strip()
        if not text:
            return min(self.max_new_tokens, self._MIN_UTTERANCE_TOKENS)
        words = len(re.findall(r"\w+", text, flags=re.UNICODE))
        chars = len(re.sub(r"\s+", "", text))
        word_s = words / self._WORDS_PER_SECOND if words else 0.0
        char_s = chars / self._CHARS_PER_SECOND if chars else 0.0
        punc = sum(unicodedata.category(c).startswith("P") for c in text)
        secs = max(word_s, char_s) + punc * 0.5 + 1.0
        toks = math.ceil(secs * self._TOKENS_PER_SECOND * self._SAFETY)
        cs = max(1, self.streaming_chunk_size)
        aligned = max(cs, math.ceil(toks / cs) * cs)
        return min(self.max_new_tokens, max(self._MIN_UTTERANCE_TOKENS, aligned))

    # ---------- 推理 ----------

    @staticmethod
    def _chunk_to_wav_sr(item):
        """generate_*_streaming 每次 yield 的可能是 (audio, sr, timing) 元组，
        也可能是带 .audio/.sample_rate 属性的对象。"""
        if isinstance(item, tuple):
            audio = item[0]
            sr = item[1] if len(item) > 1 else TARGET_SR
        else:
            audio = getattr(item, "audio", None)
            sr = getattr(item, "sample_rate", None) or TARGET_SR
        if audio is None:
            return None, None
        return np.asarray(audio, dtype=np.float32).squeeze(), int(sr)

    def _synth(self, text: str):
        """返回 (float32 波形, 采样率)，把流式分块拼起来"""
        ref_audio = self._current_ref()
        ref_text = self._current_ref_text()

        full_kwargs = dict(
            text=text,
            language=self.language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            chunk_size=self.streaming_chunk_size,
            max_new_tokens=self._estimate_max_new_tokens(text),
        )
        with torch.no_grad():
            try:
                gen = self.model.generate_voice_clone_streaming(**full_kwargs)
            except TypeError:
                # 老/新版本 kwargs 不一致时回退到最小参数集
                gen = self.model.generate_voice_clone_streaming(
                    text=text, ref_audio=ref_audio, ref_text=ref_text
                )

            chunks, sr = [], TARGET_SR
            for item in gen:
                wav, s = self._chunk_to_wav_sr(item)
                if wav is None or wav.size == 0:
                    continue
                chunks.append(wav)
                sr = s

        if not chunks:
            return np.zeros(0, dtype=np.float32), TARGET_SR
        return np.concatenate(chunks), int(sr)

    @staticmethod
    def _to_16k_int16(wav: np.ndarray, sr: int) -> np.ndarray:
        """Qwen3-TTS 出 24kHz float，管线要 16kHz int16"""
        if wav.size == 0:
            return wav.astype(np.int16)
        if sr != TARGET_SR:
            g = np.gcd(sr, TARGET_SR)
            wav = resample_poly(wav, TARGET_SR // g, sr // g)
        wav = np.clip(wav, -1.0, 1.0)
        return (wav * 32767).astype(np.int16)

    # ---------- 管线接口 ----------

    def process(self, llm_sentence):
        if isinstance(llm_sentence, tuple):
            llm_sentence, _lang_code = llm_sentence

        logger.info(f"[TTS] 收到文本: {llm_sentence!r}")

        # 空句 / 纯标点直接跳过，否则会合成出一段杂音
        if not llm_sentence or not llm_sentence.strip(" 。，！？.,!?…~"):
            logger.info("[TTS] 文本为空/纯标点，跳过")
            self.should_listen.set()
            return

        try:
            wav, sr = self._synth(llm_sentence)
            audio = self._to_16k_int16(wav, sr)
        except Exception as e:
            logger.error(f"[TTS] 合成失败: {e}", exc_info=True)
            self.should_listen.set()
            return

        logger.info(f"[TTS] 合成完成: {audio.size} 采样 ≈ {audio.size/TARGET_SR:.2f}s")
        if audio.size == 0:
            logger.warning("[TTS] 输出为空音频，跳过")
            self.should_listen.set()
            return

        # avatar 模式: 音频交给 Ditto 形象服务生成说话视频, 不本地播放
        if self.avatar_mode:
            self._send_to_avatar(audio)
            return

        # 补零到 blocksize 整数倍，避免最后一块被截断爆音
        pad = (-len(audio)) % self.blocksize
        if pad:
            audio = np.concatenate([audio, np.zeros(pad, dtype=np.int16)])

        # local 模式的播放回调要的是 int16 numpy 数组(每块正好 blocksize 个采样)，
        # 不是字节。这里 yield ndarray，与 melo_handler 一致。
        for i in range(0, len(audio), self.blocksize):
            yield audio[i : i + self.blocksize]

        # 说完了，把麦克风放回去 —— 漏掉这行 AI 说完一句就聋了
        self.should_listen.set()

    def _avatar_ssl_ctx(self):
        # 网页版下 avatar_url 是 https://127.0.0.1:8902(本机自签证书, 给浏览器用的),
        # 这里是同机后端到后端的调用, 不校验证书(不是暴露在外网的连接)
        if not self.avatar_url.startswith("https://"):
            return None
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def _send_to_avatar(self, audio: np.ndarray):
        """把整句音频(int16 16k)POST 给 Ditto 形象服务; 阻塞到视频生成完，
        再等约一个音频时长(浏览器播放期)才放回麦克风，避免边说边听。"""
        dur = len(audio) / TARGET_SR
        try:
            req = urllib.request.Request(
                self.avatar_url, data=audio.tobytes(), method="POST",
                headers={"Content-Type": "application/octet-stream"})
            urllib.request.urlopen(req, timeout=180, context=self._avatar_ssl_ctx()).read()   # 阻塞：生成完成
        except Exception as e:
            logger.error(f"[TTS] 发送到形象服务失败: {e}")
            self.should_listen.set()
            return
        # 生成已耗时约 1.3×dur；再等 dur 让浏览器把这段视频播完
        time.sleep(dur)
        self.should_listen.set()

    def _pregenerate_fillers(self):
        """合成当前角色语气的过渡语("让我想想哈"之类), 逐个POST给Ditto预生成
        视频缓存进池子; 首次在 setup() 里做一次, 切角色时(音色变了)重做一次。
        纯优化项, 失败不影响主流程(顶多没有过渡语, 退回原来的静默等待)。"""
        try:
            reset_req = urllib.request.Request(f"{self.avatar_base}/reset_fillers", data=b"", method="POST")
            urllib.request.urlopen(reset_req, timeout=10, context=self._avatar_ssl_ctx()).read()
            phrases = self._current_fillers()
            ok = 0
            for i, phrase in enumerate(phrases):
                try:
                    wav, sr = self._synth(phrase)
                    audio = self._to_16k_int16(wav, sr)
                    if audio.size == 0:
                        continue
                    req = urllib.request.Request(
                        f"{self.avatar_base}/generate_filler?index={i}", data=audio.tobytes(),
                        method="POST", headers={"Content-Type": "application/octet-stream"})
                    urllib.request.urlopen(req, timeout=60, context=self._avatar_ssl_ctx()).read()
                    ok += 1
                except Exception as e:
                    logger.warning(f"[TTS] 过渡语预生成失败(第{i}条 {phrase!r}): {e}")
            logger.info(f"[TTS] 过渡语预生成完成: {ok}/{len(phrases)}")
        except Exception as e:
            logger.warning(f"[TTS] 过渡语预生成整体失败(不致命, 退回无铺垫语): {e}")

    def cleanup(self):
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()

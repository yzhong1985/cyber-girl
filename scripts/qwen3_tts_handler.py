"""
TTS/qwen3_tts_handler.py

把 Qwen3-TTS 接进 huggingface/speech-to-speech 的 TTS handler。
放到项目的 TTS/ 目录下，然后按教程改 s2s_pipeline.py 注册。

依赖：
    git clone https://github.com/QwenLM/Qwen3-TTS && cd Qwen3-TTS && pip install -e .
    pip install soundfile scipy

⚠️ _synth() 里的推理调用是唯一需要跟官方仓库对齐的地方。
   Qwen3-TTS 开源还不久，API 可能已经变了，跑之前对一下官方 README。
"""

import logging
from threading import Event

import numpy as np
import torch
from scipy.signal import resample_poly

from baseHandler import BaseHandler

logger = logging.getLogger(__name__)

# 下游 socket 发送端写死 16kHz int16，这里必须重采样过去
TARGET_SR = 16000


class Qwen3TTSHandler(BaseHandler):
    """
    输入：LLM 吐出的一句话（str，或 (str, lang_code) 元组）
    输出：int16 PCM 分块，16kHz 单声道
    """

    def setup(
        self,
        should_listen: Event,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device: str = "cuda",
        torch_dtype: str = "float16",
        ref_audio: str = "voices/default.wav",
        language: str = "Chinese",
        blocksize: int = 512,
        registry=None,          # 传入 VoiceRegistry 就能热切换音色
    ):
        self.should_listen = should_listen
        self.device = device
        self.blocksize = blocksize
        self.language = language
        self.ref_audio = ref_audio
        self.registry = registry

        from qwen_tts import Qwen3TTSModel

        self.model = (
            Qwen3TTSModel.from_pretrained(
                model_name,
                torch_dtype=getattr(torch, torch_dtype),
            )
            .to(device)
            .eval()
        )
        logger.info(f"Qwen3-TTS loaded: {model_name} on {device}")
        self.warmup()

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

    # ---------- 推理 ----------

    def _current_ref(self):
        """热切换入口：registry 里的当前音色优先于启动参数"""
        if self.registry is not None:
            return self.registry.current_ref_audio()
        return self.ref_audio

    def _synth(self, text: str):
        """返回 (float32 波形, 采样率)"""
        with torch.no_grad():
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                voice_sample_path=self._current_ref(),
                language=self.language,
            )
        wav = np.asarray(wavs[0], dtype=np.float32).squeeze()
        return wav, int(sr)

    @staticmethod
    def _to_16k_int16(wav: np.ndarray, sr: int) -> np.ndarray:
        """Qwen3-TTS 出 24kHz float，管线要 16kHz int16"""
        if sr != TARGET_SR:
            g = np.gcd(sr, TARGET_SR)
            wav = resample_poly(wav, TARGET_SR // g, sr // g)
        wav = np.clip(wav, -1.0, 1.0)
        return (wav * 32767).astype(np.int16)

    # ---------- 管线接口 ----------

    def process(self, llm_sentence):
        lang_code = None
        if isinstance(llm_sentence, tuple):
            llm_sentence, lang_code = llm_sentence

        logger.debug(f"TTS in: {llm_sentence}")

        # 空句 / 纯标点直接跳过，否则会合成出一段杂音
        if not llm_sentence or not llm_sentence.strip(" 。，！？.,!?…~"):
            self.should_listen.set()
            return

        try:
            wav, sr = self._synth(llm_sentence)
            audio = self._to_16k_int16(wav, sr)
        except Exception as e:
            logger.error(f"TTS 合成失败: {e}")
            self.should_listen.set()
            return

        # 补零到 blocksize 整数倍，避免最后一块被截断爆音
        pad = (-len(audio)) % self.blocksize
        if pad:
            audio = np.concatenate([audio, np.zeros(pad, dtype=np.int16)])

        for i in range(0, len(audio), self.blocksize):
            yield audio[i : i + self.blocksize].tobytes()

        # 说完了，把麦克风放回去 —— 漏掉这行 AI 说完一句就聋了
        self.should_listen.set()

    def cleanup(self):
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()

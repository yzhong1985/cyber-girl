import os
import ssl
import time
import urllib.request
from threading import Thread

import torchaudio
from VAD.vad_iterator import VADIterator
from baseHandler import BaseHandler
import numpy as np
import torch
from rich.console import Console

from utils.utils import int2float
import logging

logger = logging.getLogger(__name__)

# Optional import for audio enhancement
try:
    from df.enhance import enhance, init_df
    HAS_DF = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_DF = False
    logger.warning(f"DeepFilterNet not available for audio enhancement: {e}")

console = Console()


class VADHandler(BaseHandler):
    """
    Handles voice activity detection. When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed
    to the following part.
    """

    def setup(
        self,
        should_listen,
        thresh=0.3,
        sample_rate=16000,
        min_silence_ms=1000,
        min_speech_ms=500,
        max_speech_ms=float("inf"),
        speech_pad_ms=30,
        audio_enhancement=False,
        enable_realtime_transcription=False,
        realtime_processing_pause=0.25,
        text_output_queue=None,
    ):
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.enable_realtime_transcription = enable_realtime_transcription
        self.realtime_processing_pause = realtime_processing_pause
        self.text_output_queue = text_output_queue
        # avatar 模式下, 说完话立刻推一段预生成好的过渡语("让我想想哈")占位播放,
        # 跟 TTS handler 的 avatar_mode 用同一个环境变量判断; 只是查一下现成的
        # 池子(avatar_service.py 那边已经预生成好了), 不在这里做任何生成/合成。
        avatar_url = os.environ.get("DITTO_AVATAR_URL")
        self.filler_url = (avatar_url.rsplit("/generate", 1)[0] + "/play_filler") if avatar_url else None
        self.filler_enabled = os.environ.get("AVATAR_FILLER", "1").lower() not in ("0", "false", "no", "off")
        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.audio_enhancement = audio_enhancement
        if audio_enhancement:
            if not HAS_DF:
                logger.error("Audio enhancement requested but DeepFilterNet is not available. Disabling audio enhancement.")
                self.audio_enhancement = False
            else:
                self.enhanced_model, self.df_state, _ = init_df()

        # State for progressive audio release
        self.accumulated_audio = []
        self.last_process_time = 0

        # Throttled logging state (summary once per second)
        self._last_log_time = 0.0
        self._log_chunks = 0
        self._log_speech_starts = 0
        self._log_speech_ends = 0
        self._log_progressive_yields = 0

    def process(self, audio_chunk):
        self._log_chunks += 1
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = int2float(audio_int16)

        # Check speech state BEFORE processing
        was_triggered_before = self.iterator.triggered

        vad_output = self.iterator(torch.from_numpy(audio_float32))

        # Check if speech state changed AFTER processing
        is_triggered_now = self.iterator.triggered
        if is_triggered_now and not was_triggered_before:
            self._log_speech_starts += 1
            logger.debug("Speech started")
            if self.text_output_queue:
                self.text_output_queue.put({"type": "speech_started"})

        # Log a summary once per second instead of every chunk
        now = time.time()
        if now - self._last_log_time >= 1.0:
            state = "SPEAKING" if is_triggered_now else "silent"
            logger.debug(
                f"VAD: {self._log_chunks} chunks/s | {state} | "
                f"starts={self._log_speech_starts} ends={self._log_speech_ends} progressive={self._log_progressive_yields}"
            )
            self._log_chunks = 0
            self._log_speech_starts = 0
            self._log_speech_ends = 0
            self._log_progressive_yields = 0
            self._last_log_time = now

        if self.enable_realtime_transcription:
            # Progressive mode: yield audio chunks while speaking
            yield from self._process_realtime(vad_output)
        else:
            # Original mode: yield only when speech ends
            yield from self._process_normal(vad_output)

    def _process_realtime(self, vad_output):
        """Process with real-time progressive audio release."""
        # Check if we're currently in a speech segment
        if hasattr(self.iterator, "buffer") and len(self.iterator.buffer) > 0:
            current_time = time.time()

            # Yield accumulated audio periodically while speaking
            if (current_time - self.last_process_time) >= self.realtime_processing_pause:
                array = torch.cat(self.iterator.buffer).cpu().numpy()
                duration_ms = len(array) / self.sample_rate * 1000

                if duration_ms >= self.min_speech_ms:
                    self._log_progressive_yields += 1
                    logger.debug(f"VAD: yielding progressive audio ({duration_ms:.0f}ms)")
                    # Yield with special flag to indicate this is progressive (not final)
                    yield ("progressive", array)
                    self.last_process_time = current_time

        # Handle end of speech
        if vad_output is not None and len(vad_output) != 0:
            logger.debug("VAD: end of speech detected")
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000

            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.debug(
                    f"VAD: skipping {duration_ms:.0f}ms segment (out of bounds)"
                )
            else:
                self._log_speech_ends += 1
                self.should_listen.clear()
                logger.debug("Stop listening")
                if self.text_output_queue:
                    self.text_output_queue.put({"type": "speech_stopped"})
                    logger.debug("Speech stopped - sent event")
                self._trigger_filler()
                if self.audio_enhancement:
                    array = self._apply_audio_enhancement(array)
                # Yield with final flag
                yield ("final", array)
                self.last_process_time = 0

    def _process_normal(self, vad_output):
        """Original processing: yield only when speech ends."""
        if vad_output is not None and len(vad_output) != 0:
            logger.debug("VAD: end of speech detected")
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.debug(
                    f"VAD: skipping {duration_ms:.0f}ms segment (out of bounds)"
                )
            else:
                self._log_speech_ends += 1
                self.should_listen.clear()
                logger.debug("Stop listening")
                if self.text_output_queue:
                    self.text_output_queue.put({"type": "speech_stopped"})
                    logger.debug("Speech stopped - sent event")
                self._trigger_filler()
                if self.audio_enhancement:
                    array = self._apply_audio_enhancement(array)
                yield array

    def _trigger_filler(self):
        """说完话立刻叫 Ditto 推一段预生成好的过渡语占位播放, 跟正常的
        STT→LLM→TTS→Ditto(在其它线程里)是真并行, 这里只是个近乎零耗时的
        HTTP调用(池子里随机挑一个已生成好的视频塞进播放队列, 不现场生成)。
        放子线程里 fire-and-forget, 绝不能因为这个调用卡住 VAD 自己的循环。"""
        if not (self.filler_url and self.filler_enabled):
            return

        def _fire():
            try:
                ctx = None
                if self.filler_url.startswith("https://"):
                    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                urllib.request.urlopen(self.filler_url, timeout=5, context=ctx).read()
            except Exception as e:
                logger.debug(f"[VAD] 过渡语触发失败(不致命): {e}")

        Thread(target=_fire, daemon=True).start()

    def _apply_audio_enhancement(self, array):
        """Apply audio enhancement if enabled."""
        if self.sample_rate != self.df_state.sr():
            audio_float32 = torchaudio.functional.resample(
                torch.from_numpy(array),
                orig_freq=self.sample_rate,
                new_freq=self.df_state.sr(),
            )
            enhanced = enhance(
                self.enhanced_model,
                self.df_state,
                audio_float32.unsqueeze(0),
            )
            enhanced = torchaudio.functional.resample(
                enhanced,
                orig_freq=self.df_state.sr(),
                new_freq=self.sample_rate,
            )
        else:
            enhanced = enhance(
                self.enhanced_model, self.df_state, torch.from_numpy(array)
            )
        return enhanced.numpy().squeeze()

    @property
    def min_time_to_debug(self):
        return 0.00001

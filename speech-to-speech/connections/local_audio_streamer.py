import threading
import queue as _queue
import sounddevice as sd
import numpy as np

import time
import logging

logger = logging.getLogger(__name__)

SR = 16000

# Live2D 口型驱动（可选，未启用时为空操作）
try:
    from live2d_panel import set_mouth as _set_mouth
except Exception:
    def _set_mouth(v):
        pass

_MOUTH_HOP = 0.04  # 40ms 一帧，与 SSE 推送节奏一致


def _envelope(data, sr=SR, hop=_MOUTH_HOP):
    """按 hop 计算音量包络(RMS)，逐句归一化到 0..1"""
    h = max(1, int(sr * hop))
    d = data.astype(np.float32) / 32768.0
    vals = []
    for i in range(0, len(d), h):
        seg = d[i:i + h]
        vals.append(float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0)
    mx = max(vals) if vals else 0.0
    if mx > 1e-4:
        vals = [min(1.0, (v / mx) ** 0.7) for v in vals]  # 归一化 + 轻微提亮低音量
    else:
        vals = [0.0 for _ in vals]
    return vals


def _play_with_mouth(data, stop_event):
    """播放整段音频，同时把音量包络同步推给 Live2D 口型"""
    env = _envelope(data)

    def drive():
        for v in env:
            if stop_event.is_set():
                break
            _set_mouth(v)
            time.sleep(_MOUTH_HOP)
        _set_mouth(0.0)

    t = threading.Thread(target=drive, daemon=True)
    t.start()
    try:
        sd.play(data, SR, blocking=True)
    except Exception as e:
        logger.error(f"sd.play 播放失败: {e}")
    finally:
        _set_mouth(0.0)


class LocalAudioStreamer:
    def __init__(
        self,
        input_queue,
        output_queue,
        list_play_chunk_size=512,
    ):
        self.list_play_chunk_size = list_play_chunk_size

        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        # WSL(pulse-ALSA) 专用做法：
        #  · 输入：独立的回调 InputStream（已验证能稳定采集）
        #  · 输出：常驻输出流(回调/阻塞)在此环境都不可靠(没声/断续/卡死)，
        #    改成每句话攒齐后用 sd.play(blocking=True) 播放——每次开一个短命的
        #    输出流，放完即关，这与实测能出声的方式一致。
        playing = threading.Event()

        def in_callback(indata, frames, time_info, status):
            if self.stop_event.is_set():
                return
            # 正在播放 AI 语音时不采集麦克风，避免自我回听
            if not playing.is_set() and self.output_queue.empty():
                self.input_queue.put(indata.copy())

        def writer():
            while not self.stop_event.is_set():
                try:
                    chunk = self.output_queue.get(timeout=0.1)
                except _queue.Empty:
                    playing.clear()
                    continue
                playing.set()
                buf = []
                if isinstance(chunk, np.ndarray):
                    buf.append(chunk.reshape(-1).astype(np.int16))
                # TTS 会把一句话的所有分块几乎同时塞进队列，这里一次性取空拼成整段
                while not self.output_queue.empty():
                    try:
                        c = self.output_queue.get_nowait()
                    except _queue.Empty:
                        break
                    if isinstance(c, np.ndarray):
                        buf.append(c.reshape(-1).astype(np.int16))
                if buf:
                    data = np.concatenate(buf)
                    _play_with_mouth(data, self.stop_event)

        logger.debug("Available devices:")
        logger.debug(sd.query_devices())

        in_stream = sd.InputStream(
            samplerate=SR, dtype="int16", channels=1,
            callback=in_callback, blocksize=self.list_play_chunk_size,
            latency="high",
        )
        with in_stream:
            wt = threading.Thread(target=writer, daemon=True)
            wt.start()
            logger.info("Starting local audio stream (sd.play output for WSL)")
            while not self.stop_event.is_set():
                time.sleep(0.001)
            print("Stopping recording")

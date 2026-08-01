"""
STT/transcribe_ref.py

给"新建角色"面板用：上传一段参考音频，自动转写成 ref_text（Qwen3-TTS 声音
克隆需要参考音频对应的文字稿，手动听写太麻烦）。

设计取舍：不复用管线里常驻的 WhisperSTTHandler 实例——那个模型在独立线程里
被 VAD 驱动的 STT 队列持续调用，这里要是插进去同时调 .generate()，存在跟
正常对话线程抢 GPU/并发调用同一个模型对象的风险。做成一个完全独立的函数：
用完即走，每次现load模型、转写、然后立刻 del + empty_cache，不常驻显存。
5090 上瞬时多出一份 whisper-large-v3-turbo（fp16，~1.6GB）完全扛得住，用完
就还回去，不影响管线长期显存占用。
"""

import logging

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/whisper-large-v3-turbo"


def transcribe_ref_audio(audio_path: str, model_name: str = DEFAULT_MODEL, device: str = "cuda") -> str:
    """转写参考音频，返回识别出的文字稿。失败时抛异常，调用方(面板)负责兜底
    提示用户手动填写，不在这里静默吞掉——用户需要知道转写失败了。"""
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, dtype=dtype).to(device)
    try:
        input_features = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype=dtype)
        with torch.no_grad():
            pred_ids = model.generate(input_features)
        text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        logger.info(f"[transcribe_ref] 转写完成: {text!r}")
        return text
    finally:
        del model, processor
        if device == "cuda":
            torch.cuda.empty_cache()

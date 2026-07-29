import time, numpy as np, torch, soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS
t0=time.time()
m = FasterQwen3TTS.from_pretrained("./models/qwen3-tts", device="cuda", dtype=torch.float16)
print(f"[load] {time.time()-t0:.1f}s  显存={torch.cuda.memory_allocated()/1e9:.2f}G")
t1=time.time(); chunks=[]; sr=24000
for audio, s, meta in m.generate_voice_clone_streaming(
        text="你好呀，我是小满。今天想跟我聊点什么？",
        language="Chinese",
        ref_audio="voices/xiaoman.wav",
        ref_text="I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs.",
        chunk_size=12, max_new_tokens=1536):
    chunks.append(np.asarray(audio,dtype=np.float32).squeeze()); sr=s
wav=np.concatenate(chunks)
sf.write("_tts_test.wav", wav, sr)
print(f"[synth] {time.time()-t1:.1f}s  时长={len(wav)/sr:.2f}s  sr={sr}  峰值={np.abs(wav).max():.3f}")
print("OK -> _tts_test.wav")

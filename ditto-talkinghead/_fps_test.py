import time, math, librosa, numpy as np, torch, os
from stream_pipeline_offline import StreamSDK
cfg="./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
data="./checkpoints/ditto_pytorch"
t0=time.time()
SDK=StreamSDK(cfg, data)
print(f"[load SDK] {time.time()-t0:.1f}s  显存={torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)
src="./example/image.png"; out="./tmp/result_novoice.mp4"; aud="./example/audio.wav"
SDK.setup(src, out)
audio,sr=librosa.core.load(aud, sr=16000)
num_f=math.ceil(len(audio)/16000*25)
print(f"num_f={num_f}  audio={len(audio)/16000:.1f}s  online={SDK.online_mode}", flush=True)
SDK.setup_Nd(N_d=num_f)
t1=time.time()
aud_feat=SDK.wav2feat.wav2feat(audio)
SDK.audio2motion_queue.put(aud_feat)
SDK.close()
gen=time.time()-t1
print(f"[GEN] {gen:.2f}s / {num_f} 帧 = {num_f/gen:.1f} fps  峰值显存={torch.cuda.max_memory_allocated()/1e9:.2f}G", flush=True)
os.system(f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{aud}" -map 0:v -map 1:a -c:v copy -c:a aac ./tmp/result.mp4')
print("OK -> tmp/result.mp4", flush=True)

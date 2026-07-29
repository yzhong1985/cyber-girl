import time, math, librosa, numpy as np, torch
from stream_pipeline_offline import StreamSDK
SDK=StreamSDK("./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl","./checkpoints/ditto_pytorch")
audio,sr=librosa.core.load("./example/audio.wav", sr=16000)
num_f=math.ceil(len(audio)/16000*25)
for steps in (50,25,15,10):
    SDK.setup("./example/image.png", f"./tmp/r_{steps}.mp4", sampling_timesteps=steps)
    SDK.setup_Nd(N_d=num_f)
    t=time.time()
    aud_feat=SDK.wav2feat.wav2feat(audio); SDK.audio2motion_queue.put(aud_feat); SDK.close()
    dt=time.time()-t
    print(f"[steps={steps:>2}] {dt:.1f}s / {num_f}帧 = {num_f/dt:.1f} fps  {'✅实时' if num_f/dt>=25 else '⏳'}", flush=True)

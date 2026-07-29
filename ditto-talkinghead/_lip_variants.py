import time, math, librosa, subprocess
from stream_pipeline_offline import StreamSDK
SDK=StreamSDK("./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl","./checkpoints/ditto_pytorch")
audio,sr=librosa.core.load("./example/audio.wav", sr=16000)
num_f=math.ceil(len(audio)/16000*25)
variants=[
  ("s50",       dict(max_size=768, sampling_timesteps=50)),
  ("s50_smo5",  dict(max_size=768, sampling_timesteps=50, smo_k_d=5)),
  ("s75",       dict(max_size=768, sampling_timesteps=75)),
]
for name,kw in variants:
    raw=f"./tmp/lip_{name}_raw.mp4"; out=f"./tmp/lip_{name}.mp4"
    SDK.setup("./avatar/girl.png", raw, **kw)
    SDK.setup_Nd(N_d=num_f)
    t=time.time()
    SDK.audio2motion_queue.put(SDK.wav2feat.wav2feat(audio)); SDK.close()
    dt=time.time()-t
    subprocess.run(f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "./example/audio.wav" -map 0:v -map 1:a -c:v copy -c:a aac "{out}"', shell=True)
    print(f"[{name}] {dt:.1f}s = {num_f/dt:.1f}fps -> tmp/lip_{name}.mp4", flush=True)

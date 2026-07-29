import time, math, librosa, torch
from stream_pipeline_offline import StreamSDK
SDK=StreamSDK("./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl","./checkpoints/ditto_pytorch")
audio,sr=librosa.core.load("./example/audio.wav", sr=16000)
num_f=math.ceil(len(audio)/16000*25)
import subprocess
for ms in (512,768,1024):
    out=f"./tmp/res_{ms}.mp4"
    SDK.setup("./example/image.png", out, max_size=ms, sampling_timesteps=25)
    SDK.setup_Nd(N_d=num_f)
    t=time.time()
    SDK.audio2motion_queue.put(SDK.wav2feat.wav2feat(audio)); SDK.close()
    dt=time.time()-t
    # 读实际输出分辨率
    r=subprocess.run(["ffprobe","-v","error","-select_streams","v","-show_entries","stream=width,height","-of","csv=p=0",SDK.tmp_output_path],capture_output=True,text=True).stdout.strip()
    print(f"[max_size={ms}] 分辨率={r}  {dt:.1f}s = {num_f/dt:.1f} fps  {'✅实时' if num_f/dt>=25 else '⏳'}", flush=True)

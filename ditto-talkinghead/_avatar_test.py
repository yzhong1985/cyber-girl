import time, math, librosa, torch, subprocess
from stream_pipeline_offline import StreamSDK
SDK=StreamSDK("./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl","./checkpoints/ditto_pytorch")
audio,sr=librosa.core.load("./example/audio.wav", sr=16000)
num_f=math.ceil(len(audio)/16000*25)
out="./tmp/avatar_test.mp4"
t=time.time()
SDK.setup("./avatar/girl.png", out, max_size=768, sampling_timesteps=25)
print(f"[注册人脸] {time.time()-t:.1f}s", flush=True)
SDK.setup_Nd(N_d=num_f)
t2=time.time()
SDK.audio2motion_queue.put(SDK.wav2feat.wav2feat(audio)); SDK.close()
dt=time.time()-t2
r=subprocess.run(["ffprobe","-v","error","-select_streams","v","-show_entries","stream=width,height","-of","csv=p=0",SDK.tmp_output_path],capture_output=True,text=True).stdout.strip()
print(f"[生成] {dt:.1f}s / {num_f}帧 = {num_f/dt:.1f} fps  分辨率={r}", flush=True)
subprocess.run(f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "./example/audio.wav" -map 0:v -map 1:a -c:v copy -c:a aac "{out}"', shell=True)
print("OK ->", out, flush=True)

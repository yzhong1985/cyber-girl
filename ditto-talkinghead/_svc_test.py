import time, urllib.request, numpy as np, librosa, json
BASE="http://127.0.0.1:8902"
# 等服务就绪
for i in range(120):
    try:
        urllib.request.urlopen(BASE+"/idle.png", timeout=3); print("服务就绪", flush=True); break
    except Exception: time.sleep(2)
else:
    print("服务未就绪，退出"); raise SystemExit
# 示例音频 -> int16 16k PCM
audio,_=librosa.core.load("./example/audio.wav", sr=16000)
pcm=(np.clip(audio,-1,1)*32767).astype(np.int16).tobytes()
print(f"POST /generate  ({len(audio)/16000:.1f}s 音频)...", flush=True)
t=time.time()
req=urllib.request.Request(BASE+"/generate", data=pcm, method="POST",
                           headers={"Content-Type":"application/octet-stream"})
r=json.loads(urllib.request.urlopen(req, timeout=120).read())
print(f"生成完成 用时{time.time()-t:.1f}s ->", r, flush=True)

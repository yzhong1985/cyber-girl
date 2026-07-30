"""对比 warp_network 用 TensorRT 引擎(本机新编译插件) vs torch.compile(PyTorch) 的稳态fps。
每个 cfg 都跑 2 轮(第1轮吸收编译/context创建开销, 取第2轮稳态fps), 与 avatar_service.py
的 setup/setup_Nd/gen/close 调用方式一致。

用法:
  .venv/bin/python _fps_test_trtwarp.py --cfg onnx --compile-warp   # 当前基线: onnx混合+torch.compile warp
  .venv/bin/python _fps_test_trtwarp.py --cfg trtwarp               # 新: warp_network 换本机TRT引擎
"""
import argparse, time, math, os
import librosa, torch
from stream_pipeline_offline import StreamSDK

CFG_BY_NAME = {
    "onnx": "./checkpoints/ditto_cfg/v0.4_hubert_cfg_onnx.pkl",
    "trtwarp": "./checkpoints/ditto_cfg/v0.4_hubert_cfg_onnx_trtwarp_local.pkl",
}

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", choices=list(CFG_BY_NAME), required=True)
ap.add_argument("--compile-warp", action="store_true")
ap.add_argument("--compile-lmdm", action="store_true")
args = ap.parse_args()

cfg = CFG_BY_NAME[args.cfg]
data = "./checkpoints/ditto_pytorch"
src = "./example/image.png"
aud = "./example/audio.wav"

t0 = time.time()
SDK = StreamSDK(cfg, data)
print(f"[load SDK cfg={args.cfg}] {time.time()-t0:.1f}s  显存={torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)

if args.compile_warp:
    print("[compile] warp_network torch.compile(reduce-overhead)...", flush=True)
    SDK.warp_f3d.warp_net.model = torch.compile(SDK.warp_f3d.warp_net.model, mode="reduce-overhead")

if args.compile_lmdm:
    print("[compile] lmdm torch.compile(default)...", flush=True)
    motion_decoder = SDK.audio2motion.lmdm.model.model
    motion_decoder.forward = torch.compile(motion_decoder.forward, mode="default")

audio, sr = librosa.core.load(aud, sr=16000)
num_f = math.ceil(len(audio) / 16000 * 25)
print(f"num_f={num_f}  audio={len(audio)/16000:.1f}s", flush=True)

os.makedirs("./tmp", exist_ok=True)
for i in range(2):
    out = f"./tmp/result_trtwarp_test_{i}.mp4"
    SDK.setup(src, out)
    SDK.setup_Nd(N_d=num_f)
    t1 = time.time()
    aud_feat = SDK.wav2feat.wav2feat(audio)
    SDK.audio2motion_queue.put(aud_feat)
    SDK.close()
    gen = time.time() - t1
    print(f"[round {i}] {gen:.2f}s / {num_f} 帧 = {num_f/gen:.1f} fps  峰值显存={torch.cuda.max_memory_allocated()/1e9:.2f}G", flush=True)

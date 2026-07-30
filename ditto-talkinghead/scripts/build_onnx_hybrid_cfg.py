"""
生成"混合 ONNX 后端"配置: 把耗时的几个核心模型换成 onnxruntime CUDA 执行。
warp_network 和 lmdm(audio2motion 扩散模型)仍留 PyTorch —— 不是因为它们不能用
onnx, 而是因为 avatar_service.py 会在加载后给它们各自套一层 torch.compile,
比 onnxruntime 更快(warp_network 的 onnx 版还有 GridSample3D 自定义算子的
限制, 标准 onnxruntime 跑不了)。

实测(RTX 5090, 1.5s 测试音频, 稳态fps):
  12.5  纯PyTorch
→ 15.0  本混合(4模型→onnx, warp+lmdm仍PyTorch)                    (2026-07-29)
→ 19.5  + torch.compile(warp_network, reduce-overhead)             (2026-07-29)
→ 31.9  + torch.compile(lmdm.MotionDecoder.forward, default模式)   (2026-07-30, 当前部署)
  (lmdm 试过 reduce-overhead 模式, 和其内部 rotary embedding 的张量缓存冲突崩溃,
   报 "CUDAGraphs 输出被后续运行覆写"; 改用不开 CUDA Graph 的 default 模式规避,
   提速反而更明显。若想榨干这部分, 需要改 Ditto 源码给 rotary embedding 那处
   补 .clone()/cudagraph_mark_step_begin(), 目前未做。)
  另试过把 warp_network 换成本机重编译的 TensorRT 引擎(third_party/
  grid-sample3d-trt-plugin, 见该目录 README)替代 torch.compile: 插件本身验证正确
  (跟 F.grid_sample 的差异在 FP16 精度噪声内), 但 fps 跟 torch.compile 打平
  (~19.6~20.3, 无提速), 且多一层插件编译依赖, 未采用。
逐帧像素差均值 <1.0/255(浮点计算顺序噪声, 无感知差异)。

用法: 在 ditto-talkinghead 目录、Ditto 的 py3.10 venv 下运行一次即可:
    .venv/bin/python scripts/build_onnx_hybrid_cfg.py
生成 checkpoints/ditto_cfg/v0.4_hubert_cfg_onnx.pkl (随包保留, 不用每次重跑)。
"""
import pickle
import os

CFG_DIR = "./checkpoints/ditto_cfg"
ONNX_DIR = os.path.abspath("./checkpoints/ditto_onnx")

SRC_CFG = os.path.join(CFG_DIR, "v0.4_hubert_cfg_pytorch.pkl")
DST_CFG = os.path.join(CFG_DIR, "v0.4_hubert_cfg_onnx.pkl")

# 换成 onnx 的核心模型(warp_network / lmdm 不在此列, 见上方说明)
SWAP_ONNX = {
    "appearance_extractor_cfg": "appearance_extractor.onnx",
    "motion_extractor_cfg": "motion_extractor.onnx",
    "stitch_network_cfg": "stitch_network.onnx",
    "decoder_cfg": "decoder.onnx",
}


def main():
    with open(SRC_CFG, "rb") as f:
        cfg = pickle.load(f)

    for key, fname in SWAP_ONNX.items():
        p = os.path.join(ONNX_DIR, fname)
        assert os.path.isfile(p), f"缺少 onnx 权重: {p}"
        cfg["base_cfg"][key]["model_path"] = p

    # warp_network / lmdm: 留 PyTorch 权重(相对路径), avatar_service.py 加载后会各自 torch.compile
    cfg["base_cfg"]["warp_network_cfg"]["model_path"] = "models/warp_network.pth"
    cfg["audio2motion_cfg"]["model_path"] = "models/lmdm_v0.4_hubert.pth"

    with open(DST_CFG, "wb") as f:
        pickle.dump(cfg, f)

    print(f"已生成: {DST_CFG}")
    print("  换 onnx:", list(SWAP_ONNX.keys()))
    print("  仍 PyTorch(会被 avatar_service.py torch.compile): warp_network, lmdm")
    print("  仍 PyTorch(本来就够快, 不受影响): 各辅助检测/关键点模型(本来就是onnx)")


if __name__ == "__main__":
    main()

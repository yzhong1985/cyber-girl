"""
生成"混合 ONNX 后端"配置: 把耗时的几个核心模型换成 onnxruntime CUDA 执行,
warp_network 仍留 PyTorch(其 onnx 版本用到自定义算子 GridSample3D, 标准
onnxruntime 不支持, 仓库里是靠一个预编译的 TensorRT 插件跑的; 若以后装好
支持该算子的 TensorRT, 可以把 warp_network 也换成 onnx)。

实测(2026-07-29, RTX 5090): 生成阶段 12.5fps(纯PyTorch) → 15.0fps(本混合),
约 +20%; 逐帧像素差均值 1.2/255(浮点计算顺序噪声, 无感知差异)。

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

# 换成 onnx 的核心模型(GridSample3D 卡住的 warp_network 不在此列)
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

    lmdm_path = os.path.join(ONNX_DIR, "lmdm_v0.4_hubert.onnx")
    assert os.path.isfile(lmdm_path), f"缺少 onnx 权重: {lmdm_path}"
    cfg["audio2motion_cfg"]["model_path"] = lmdm_path

    # warp_network: onnx 版本因 GridSample3D 算子跑不了(见本文件头注释), 留 PyTorch
    cfg["base_cfg"]["warp_network_cfg"]["model_path"] = "models/warp_network.pth"

    with open(DST_CFG, "wb") as f:
        pickle.dump(cfg, f)

    print(f"已生成: {DST_CFG}")
    print("  换 onnx:", list(SWAP_ONNX.keys()) + ["lmdm(audio2motion)"])
    print("  仍 PyTorch: warp_network(GridSample3D 算子限制) + 各辅助检测/关键点模型(本来就是onnx,不受影响)")


if __name__ == "__main__":
    main()

import os

import numpy as np
import torch
from ..utils.load_model import load_model

# warp_network 的 onnx/TRT 图里有自定义算子 GridSample3D, 标准 TensorRT 不支持,
# 要跑 TRT 引擎(.engine)必须先把这个插件 .so 加载进插件注册表; load_model()->
# TRTWrapper 默认不传 plugin_file_list, 所以这里单独处理 .engine 情形。
_GRIDSAMPLE3D_PLUGIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party", "grid-sample3d-trt-plugin", "build", "libgrid_sample_3d_plugin.so",
)


class WarpNetwork:
    def __init__(self, model_path, device="cuda"):
        if model_path.endswith(".engine") or model_path.endswith(".trt"):
            from ..utils.tensorrt_utils import TRTWrapper
            plugin_files = [_GRIDSAMPLE3D_PLUGIN] if os.path.isfile(_GRIDSAMPLE3D_PLUGIN) else []
            self.model = TRTWrapper(model_path, plugin_file_list=plugin_files)
            self.model_type = "tensorrt"
        else:
            kwargs = {
                "module_name": "WarpingNetwork",
            }
            self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

    def __call__(self, feature_3d, kp_source, kp_driving):
        """
        feature_3d: np.ndarray, shape (1, 32, 16, 64, 64)
        kp_source | kp_driving: np.ndarray, shape (1, 21, 3)
        """
        if self.model_type == "onnx":
            pred = self.model.run(None, {"feature_3d": feature_3d, "kp_source": kp_source, "kp_driving": kp_driving})[0]
        elif self.model_type == "tensorrt":
            self.model.setup({"feature_3d": feature_3d, "kp_source": kp_source, "kp_driving": kp_driving})
            self.model.infer()
            pred = self.model.buffer["out"][0].copy()
        elif self.model_type == 'pytorch':
            with torch.no_grad(), torch.autocast(device_type=self.device[:4], dtype=torch.float16, enabled=True):
                pred = self.model(
                    torch.from_numpy(feature_3d).to(self.device), 
                    torch.from_numpy(kp_source).to(self.device), 
                    torch.from_numpy(kp_driving).to(self.device)
                ).float().cpu().numpy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return pred

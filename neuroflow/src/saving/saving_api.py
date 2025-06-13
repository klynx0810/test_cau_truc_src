import json
import h5py
import numpy as np
from .saving_lib import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Activation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..models.model import Model


def save_model_to_h5(model: "Model", filepath: str):
    """
    Lưu toàn bộ model (cấu trúc + trọng số) vào file .h5
    """
    with h5py.File(filepath, "w") as f:
        # 1. Lưu config dưới dạng JSON
        config = []
        for layer in model.layers:
            if hasattr(layer, "get_config"):
                layer_config = layer.get_config()
                layer_config["class"] = layer.__class__.__name__
                config.append(layer_config)
        f.attrs["model_config"] = json.dumps(config)

        # 2. Lưu trọng số
        for i, layer in enumerate(model.layers):
            if hasattr(layer, "params"):
                group = f.create_group(f"layer_{i}")
                for key, val in layer.params.items():
                    group.create_dataset(key, data=val)


def load_model_from_h5(filepath: str, model_class: "Model"):
    """
    Tải model từ file .h5 đã lưu.
    Cần truyền vào class gốc, ví dụ: Sequential
    """
    with h5py.File(filepath, "r") as f:
        config = json.loads(f.attrs["model_config"])
        model: "Model" = model_class()

        for layer_cfg in config:
            layer_class = layer_cfg.pop("class")
            
            if layer_class == "Conv2D":
                model.add(Conv2D(**layer_cfg))
            elif layer_class == "Dense":
                model.add(Dense(**layer_cfg))
            elif layer_class == "Flatten":
                model.add(Flatten(**layer_cfg))
            elif layer_class == "Dropout":
                model.add(Dropout(**layer_cfg))
            elif layer_class == "MaxPooling2D":
                model.add(MaxPooling2D(**layer_cfg))
            elif layer_class == "Activation":
                model.add(Activation(layer_cfg["activation"]))
            else:
                raise ValueError(f"Unknown layer class: {layer_class}")

        # Load trọng số
        for i, layer in enumerate(model.layers):
            if hasattr(layer, "params"):
                for key in layer.params:
                    layer.params[key] = f[f"layer_{i}"][key][...]

    return model

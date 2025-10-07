import importlib
from dataclasses import dataclass
from typing import Union, Tuple, Optional
from stage1 import RAE
import torch.nn as nn
from omegaconf import OmegaConf
import torch

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    model = get_obj_from_str(config["target"])(**config.get("params", dict()))
    ckpt_path = config.get("ckpt", None)
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    return model


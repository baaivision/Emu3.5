import os.path as osp

from omegaconf import OmegaConf
import torch

from .ibq import IBQ


def build_vision_tokenizer(type, model_path, device="cuda:0", **kwargs):
    match type:
        case "ibq":
            cfg = OmegaConf.load(osp.join(model_path, "config.yaml"))
            tokenizer = IBQ(**cfg)
            ckpt = torch.load(osp.join(model_path, "model.ckpt"), map_location="cpu")
            tokenizer.load_state_dict(ckpt)
            tokenizer.eval().to(device)
            return tokenizer
        case _:
            raise NotImplementedError(f"Unsupported vision tokenizer type: {type}")


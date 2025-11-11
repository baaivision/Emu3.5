# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import threading
import time
from typing import Any, Dict, Generator, List, Optional
from pathlib import Path
from PIL import Image
import torch

from src.utils.input_utils import build_image
from src.utils.model_utils import build_emu3p5
from src.utils.generation_utils import generate, multimodal_decode


class ModelRuntime:
    _singleton: Optional["ModelRuntime"] = None

    _sampling_keys = [
        "top_p", "top_k", "temperature", "num_beams", "max_new_tokens",
        "min_new_tokens", "repetition_penalty", "do_sample"
    ]

    @classmethod
    def instance(cls) -> "ModelRuntime":
        if cls._singleton is None:
            cls._singleton = ModelRuntime()
        return cls._singleton

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.vq_model = None

        self.cfg_module: Optional[Any] = None
        self.runtime_persist_cfg: Dict = {}

        self._device: Optional[torch.device] = None
        self._save_dir: Optional[str] = None
        self._stop_event = threading.Event()

        self.history: List = []

    def _load_cfg_module(self, cfg_path: str):
        import importlib.util
        cfg_path = os.path.abspath(cfg_path)
        spec = importlib.util.spec_from_file_location(Path(cfg_path).stem, cfg_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def initialize(self, cfg_path: str, save_dir: str,
                   device_str: Optional[str] = None) -> str:

        if self.model is not None:
            return "✅ The model is ready (pre-loaded)"

        cfg = self._load_cfg_module(cfg_path)

        device = torch.device(device_str) if device_str else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model, self.tokenizer, self.vq_model = build_emu3p5(
            cfg.model_path,
            cfg.tokenizer_path,
            cfg.vq_path,
            vq_type=getattr(cfg, "vq_type", "ibq"),
            model_device=getattr(cfg, "hf_device", device),
            vq_device=getattr(cfg, "vq_device", device),
            **getattr(cfg, "diffusion_decoder_kwargs", {}),
        )

        cfg.special_token_ids = {
            k: self.tokenizer.convert_tokens_to_ids(v)
            for k, v in cfg.special_tokens.items()
        }

        self.runtime_persist_cfg = {
            "special_token_ids": cfg.special_token_ids
        }

        save_dir = getattr(cfg, "save_path", save_dir)
        os.makedirs(save_dir, exist_ok=True)

        self.cfg_module = cfg
        self.cfg_module.streaming = True
        self._device = device
        self._save_dir = save_dir

        return f"✅ The model has been loaded onto {device}, and the output directory is: {save_dir}"

    # ---------------- Switch to "config" mode without reloading the model. -----------------
    def update_sampling_config(self, mode: str, target_height: int = None, target_width: int = None):
        config_map = {
            "howto": "configs/example_config_visual_guidance.py",
            "story": "configs/example_config_visual_narrative.py",
            "t2i": "configs/example_config_t2i.py",
            "x2i": "configs/example_config_x2i.py",
            "default": "configs/config.py",
        }

        cfg_file = config_map.get(mode, "configs/config.py")
        cfg = self._load_cfg_module(cfg_file)

        cfg.special_token_ids = {
            k: self.tokenizer.convert_tokens_to_ids(v)
            for k, v in cfg.special_tokens.items()
        }

        self.runtime_persist_cfg = {
            "special_token_ids": cfg.special_token_ids
        }
        
        save_dir = getattr(cfg, "save_path", self._save_dir)
        os.makedirs(save_dir, exist_ok=True)

        if mode == 't2i' and target_height is not None and target_width is not None:
            cfg.target_height = target_height
            cfg.target_width = target_width

        self.cfg_module = cfg
        self.cfg_module.streaming = True
        self._save_dir = save_dir

        print(f"[sampling updated] mode={mode}, model reused ✅, output dir: {save_dir}")

    def request_stop(self): self._stop_event.set()
    def reset_stop(self): self._stop_event.clear()

    def encode_and_set_prompt(self, sample: Dict[str, Any]):

        input_ids, unconditional_ids = self.encode_prompt(sample)
        self.history = [(input_ids, unconditional_ids)]

        session_dir = os.path.join(self._save_dir, f"session_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(session_dir, exist_ok=True)
        self._current_session_dir = session_dir

        # Save the user's text input
        user_text = sample.get("text", "")
        with open(os.path.join(session_dir, "task.txt"), "w", encoding="utf-8") as f:
            f.write(user_text)

        # Save the user's image input
        for idx, p in enumerate(sample.get("images", [])):
            try:
                Image.open(p).save(os.path.join(session_dir, f"task_image_{idx}.png"))
            except:
                pass

    def encode_prompt(self, sample: Dict[str, Any]):
        cfg = self.cfg_module
        text_prompt = sample.get("text", "")
        images = sample.get("images", [])

        unc_prompt, template = cfg.build_unc_and_template(cfg.task_type, with_image=bool(images))

        if images:
            image_str = "".join(
                build_image(Image.open(p).convert("RGB"), cfg, self.tokenizer, self.vq_model)
                for p in images
            )
            prompt = template.format(question=text_prompt)
            print(f"prompt: {prompt}")
            print(f"unc_prompt: {unc_prompt}")
            prompt = prompt.replace("<|IMAGE|>", image_str)
            unc_prompt = unc_prompt.replace("<|IMAGE|>", image_str)
        else:
            prompt = template.format(question=text_prompt)
            print(f"prompt: {prompt}")
            print(f"unc_prompt: {unc_prompt}")

        return (
            self.tokenizer.encode(prompt, return_tensors="pt").to(self._device),
            self.tokenizer.encode(unc_prompt, return_tensors="pt").to(self._device)
        )

    # ---------------- Streaming: save model's output text & image -----------------
    def stream_events(self, text_chunk_tokens: int = 64) -> Generator[Dict[str, Any], None, None]:

        input_ids, unconditional_ids = self.history[-1]
        session_dir = getattr(self, "_current_session_dir", self._save_dir)

        img_idx, text_idx = 0, 0
        text_buffer = ""

        for ev in generate(self.cfg_module, self.model, self.tokenizer,
                           input_ids, unconditional_ids, None, True):

            if self._stop_event.is_set():
                yield {"type": "text", "text": "🛑 Generation has been stopped."}
                self.reset_stop()
                break

            # ---------------- Streaming text event ----------------
            if ev["type"] == "text":
                txt = ev["text"]
                yield {"type": "text", "text": txt}
                text_buffer += txt

                if self.cfg_module.special_tokens['EOC'] in text_buffer:
                    with open(os.path.join(session_dir, f"gen_text_{text_idx}.txt"),
                                "w", encoding="utf-8") as f:
                        f.write(text_buffer)
                    text_idx += 1
                    text_buffer = ""

            # ---------------- Streaming image event ----------------
            elif ev["type"] == "image":
                image_token_str = ev["image"]
                mm_out = multimodal_decode(image_token_str, self.tokenizer, self.vq_model)
                assert len(mm_out) == 1 and "image" in mm_out[0]
                image = mm_out[0][-1]
                img_path = os.path.join(session_dir, f"gen_img_{img_idx}.png")
                image.save(img_path)
                img_idx += 1

                yield {"type": "image", "paths": [img_path]}
            
            elif ev["type"] == "broken_image":
                yield {"type": "broken_image", "broken_image": ""}
            
            else:
                pass


    @property
    def save_dir(self): 
        return self._save_dir
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib as imp
import io
import os
import os.path as osp
from pathlib import Path
import random

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from src.utils.model_utils import build_emu3p5
from src.utils.generation_utils import generate, multimodal_decode
from src.utils.painting_utils import ProtoWriter
from src.utils.input_utils import build_image, smart_resize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="", type=str)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--worker_id", default=0, type=int)
    args = parser.parse_args()
    return args


def inference(
    cfg,
    model,
    tokenizer,
    vq_model,
):
    sampling_params = cfg.sampling_params
    save_path = cfg.save_path


    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/proto", exist_ok=True)
    proto_writer = ProtoWriter()

    for name, question in tqdm(cfg.prompts, total=len(cfg.prompts)):
        if osp.exists(f"{save_path}/proto/{name}.pb"):
            print(f"result already exists, skipping {name}")
            continue

        torch.cuda.empty_cache()

        reference_image = None
        if not isinstance(question, str):
            reference_image = Image.open(question["reference_image"])
            question = question["prompt"]

        proto_writer.clear()
        proto_writer.extend([["question", question]])
        if reference_image is not None:
            proto_writer.extend([["reference_image", reference_image]])

        success = True
        prompt = cfg.template.format(question=question)

        if reference_image is not None:
            image_str = build_image(reference_image, cfg, tokenizer, vq_model)
            prompt = prompt.replace("<|IMAGE|>", image_str)
            unc_prompt = cfg.unc_prompt.replace("<|IMAGE|>", image_str)
        else:
            unc_prompt = cfg.unc_prompt

        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

        if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
            BOS = torch.Tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([BOS, input_ids], dim=1)

        unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

        if hasattr(cfg, "img_unc_prompt"):
            full_unc_ids = tokenizer.encode(cfg.img_unc_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        else:
            full_unc_ids = None

        for result in generate(cfg, model, tokenizer, input_ids, unconditional_ids, full_unc_ids):
            try:
                result = tokenizer.decode(result, skip_special_tokens=False)
                mm_out = multimodal_decode(result, tokenizer, vq_model)
                proto_writer.extend(mm_out)
            except Exception as e:
                success = False
                break

        if not success:
            continue

        proto_writer.save(f"{save_path}/proto/{name}.pb")


def main():
    args = parse_args()
    cfg_name = Path(args.cfg).stem
    cfg_package = Path(args.cfg).parent.__str__().replace("/", ".")
    cfg = imp.import_module(f".{cfg_name}", package=cfg_package)

    rank, world_size = args.worker_id, args.num_workers

    cfg.rank = rank
    cfg.world_size = world_size

    if isinstance(cfg.prompts, dict):
        cfg.prompts = [(n, p) for n, p in cfg.prompts.items()]
    else:
        cfg.prompts = [(f"story_{idx:03d}", p) for idx, p in enumerate(cfg.prompts)]

    cfg.prompts = [(n, p) for n, p in cfg.prompts if not osp.exists(f"{cfg.save_path}/proto/{n}.pb")]
    cfg.prompts = cfg.prompts[rank::world_size]
    cfg.num_prompts = len(cfg.prompts)

    hf_device, vq_device = cfg.hf_device, cfg.vq_device

    model, tokenizer, vq_model = build_emu3p5(
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.vq_path,
        vq_type=cfg.vq_type,
        model_device=hf_device,
        vq_device=vq_device,
        **getattr(cfg, "diffusion_decoder_kwargs", {}),
    )

    cfg.special_token_ids = {}
    for k, v in cfg.special_tokens.items():
        cfg.special_token_ids[k] = tokenizer.encode(v)[0]

    random.seed(cfg.seed + rank)

    inference(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        vq_model=vq_model,
    )


if __name__ == "__main__":
    main()

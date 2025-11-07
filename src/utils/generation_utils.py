# -*- coding: utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import re
from PIL import Image

import numpy as np
import torch
from transformers import GenerationConfig
from transformers.generation import LogitsProcessorList

from .logits_processor import UnbatchedClassifierFreeGuidanceLogitsForVisualTokenWithDifferentialTopKProcessor


@torch.no_grad()
def generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
    full_unconditional_ids=None,
    force_same_image_size=True,
):
    if cfg.streaming:
        raise NotImplementedError("Not supported streaming generation")
        # yield from streaming_generate(cfg, model, tokenizer, input_ids, unconditional_ids, full_unconditional_ids)
    else:
        yield non_streaming_generate(cfg, model, tokenizer, input_ids, unconditional_ids, full_unconditional_ids, force_same_image_size)


def streaming_generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
    full_unconditional_ids=None,
):
    pass


def non_streaming_generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
    full_unconditional_ids=None,
    force_same_image_size=True,
):
    input_ids_len = input_ids.shape[1]

    logits_processor = LogitsProcessorList()
    logits_processor.append(
        build_logits_processor(
            cfg,
            unconditional_ids,
            model,
            tokenizer,
            full_unconditional_ids,
            force_same_image_size=force_same_image_size,
        )
    )

    generation_config = GenerationConfig(
        **cfg.sampling_params,
        pad_token_id=cfg.special_token_ids["PAD"],
        eos_token_id=cfg.special_token_ids["EOS"],
    )

    token_ids = model.generate(
        input_ids,
        generation_config,
        logits_processor=logits_processor,
    )

    gen_token_ids = token_ids[:, input_ids_len:]
    return gen_token_ids[0].detach().cpu().numpy()


def build_logits_processor(
    cfg,
    unconditional_ids,
    model,
    tokenizer,
    full_unconditional_ids=None,
    force_same_image_size=True,
):
    logits_processor = UnbatchedClassifierFreeGuidanceLogitsForVisualTokenWithDifferentialTopKProcessor(
        guidance_scale=cfg.classifier_free_guidance,
        unconditional_ids=unconditional_ids,
        full_unconditional_ids=full_unconditional_ids,
        model=model,
        tokenizer=tokenizer,
        unconditional_type=cfg.unconditional_type,
        target_height=getattr(cfg, "target_height", None),
        target_width=getattr(cfg, "target_width", None),
        image_cfg_scale=getattr(cfg, "image_cfg_scale", 1.0),
        use_differential_sampling=cfg.sampling_params["use_differential_sampling"],
        text_top_k=cfg.sampling_params["text_top_k"],
        text_top_p=cfg.sampling_params["text_top_p"],
        text_temperature=cfg.sampling_params["text_temperature"],
        image_top_k=cfg.sampling_params["image_top_k"],
        image_top_p=cfg.sampling_params["image_top_p"],
        image_temperature=cfg.sampling_params["image_temperature"],
        force_same_image_size=force_same_image_size,
    )

    return logits_processor


@torch.no_grad()
def multimodal_decode(
    outputs,
    tokenizer,
    vision_tokenizer,
):
    outputs = outputs.replace("<|extra_101|>", "").replace("<|extra_204|>", "")
    pattern = re.compile(
        rf"({re.escape(tokenizer.bog_token)}.*?{re.escape(tokenizer.eog_token)}|"
        rf"{re.escape(tokenizer.boc_token)}.*?{re.escape(tokenizer.eoc_token)}|"
        rf"{re.escape(tokenizer.boi_token)}.*?{re.escape(tokenizer.eoi_token)})",
        re.DOTALL
    )

    multimodal_output = []
    chunks = re.split(pattern, outputs)
    for c in chunks:
        if len(c) == 0:
            continue

        if tokenizer.boi_token in c and tokenizer.eoi_token in c:
            image = decode_image(c, tokenizer, vision_tokenizer)
            if image is not None:
                multimodal_output.append(("image", image))
        elif tokenizer.bog_token in c and tokenizer.eog_token in c:
            multimodal_output.append(
                ("global_cot", c.replace(tokenizer.bog_token, "").replace(tokenizer.eog_token, ""))
            )
        elif tokenizer.boc_token in c and tokenizer.eoc_token in c:
            multimodal_output.append(
                ("image_cot", c.replace(tokenizer.boc_token, "").replace(tokenizer.eoc_token, ""))
            )
        # exclude incomplete image
        elif tokenizer.boi_token not in c and len(c.strip()) > 0:
            multimodal_output.append(("text", c))

    return multimodal_output


def decode_image(image_string, tokenizer, vision_tokenizer):
    image = []
    image_rows = re.split(re.escape(tokenizer.eol_token), image_string)
    for r in image_rows:
        token_ids = re.findall(r"<\|visual token (\d+)\|>", r)
        if len(token_ids) > 0:
            row_token = [int(m) for m in token_ids]
            image.append(row_token)

    try:
        image = torch.tensor(image, dtype=torch.long, device=next(iter(vision_tokenizer.parameters())).device)
        h, w = image.shape
        image = vision_tokenizer.decode_code(image[None], shape=(1, h, w, 256)).float()
        image = image[0].permute(1, 2, 0)
        image = Image.fromarray(((image + 1.0) * 127.5).clamp(0, 255).detach().cpu().numpy().astype(np.uint8))
        return image
    except Exception as ex:
        print(f"decode image failed {ex}")
        return None

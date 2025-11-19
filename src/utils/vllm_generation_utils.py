# -*- coding: utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from vllm import SamplingParams

@torch.no_grad()
def generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
):

    if getattr(cfg, "streaming", False):
        raise ValueError("Streaming generation is not supported in VLLM yet.")
    else:
        yield non_streaming_generate(
            cfg, model, tokenizer, input_ids, unconditional_ids,
        )


def non_streaming_generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
):
    inputs = {
        "prompt_token_ids": input_ids.tolist()[0],
        "uncond_prompt_token_ids": unconditional_ids.tolist()[0]
    }

    extra_args = {
        "guidance_scale": cfg.classifier_free_guidance,
        "text_top_k": cfg.sampling_params["text_top_k"],
        "text_top_p": cfg.sampling_params["text_top_p"],
        "text_temperature": cfg.sampling_params["text_temperature"],
        "visual_top_k": cfg.sampling_params["image_top_k"],
        "visual_top_p": cfg.sampling_params["image_top_p"],
        "visual_temperature": cfg.sampling_params["image_temperature"],
        "width": getattr(cfg, "target_width", None),
        "height": getattr(cfg, "target_height", None),
        "area": cfg.image_area if getattr(cfg, "target_width", None) else None,
    }
    if cfg.task_type in ["t2i", "x2i"]:
        stop_token_ids = tokenizer.encode("<|image end|>")
    else:
        stop_token_ids = tokenizer.encode("<|extra_204|>")
    sampling_params = SamplingParams(
        top_k=cfg.sampling_params["top_k"],
        top_p=cfg.sampling_params["top_p"],
        temperature=cfg.sampling_params["temperature"],
        max_tokens=cfg.sampling_params["max_new_tokens"],
        detokenize=False,
        extra_args=extra_args,
        stop_token_ids=stop_token_ids,
    )

    results = model.generate(inputs, sampling_params=sampling_params)
    gen_token_ids = np.array(results[0].outputs[0].token_ids)

    return gen_token_ids

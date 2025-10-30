# -*- coding: utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import torch
from transformers import AutoTokenizer

from ..emu3p5 import Emu3ForCausalLM, Emu3Config
from ..vision_tokenizer import build_vision_tokenizer

def build_emu3p5(
    model_path,
    tokenizer_path,
    vq_path,
    vq_type="ibq",
    model_device="auto",
    vq_device="cuda:0",
    **kwargs,
):
    if isinstance(model_device, int):
        device_map = f"cuda:{model_device}"
    else:
        device_map = model_device

    print(device_map)

    # MLLM
    model_config = Emu3Config.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = Emu3ForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    
    # text tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        special_tokens_file=osp.join(tokenizer_path, "emu3_vision_tokens.txt"),
        trust_remote_code=True,
    )
    tokenizer.bos_token = "<|extra_203|>"
    tokenizer.eos_token = "<|extra_204|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eol_token = "<|extra_200|>"
    tokenizer.eof_token = "<|extra_201|>"
    tokenizer.tms_token = "<|extra_202|>"
    tokenizer.img_token = "<|image token|>"
    tokenizer.boi_token = "<|image start|>"
    tokenizer.eoi_token = "<|image end|>"
    tokenizer.bss_token = "<|extra_100|>"
    tokenizer.ess_token = "<|extra_101|>"
    tokenizer.bog_token = "<|extra_60|>"
    tokenizer.eog_token = "<|extra_61|>"
    tokenizer.boc_token = "<|extra_50|>"
    tokenizer.eoc_token = "<|extra_51|>"

    # vq tokenizer
    vq_model = build_vision_tokenizer(vq_type, vq_path, device=vq_device, **kwargs)

    return model, tokenizer, vq_model


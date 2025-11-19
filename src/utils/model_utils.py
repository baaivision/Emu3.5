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
        # attn_implementation="eager", # if you cann't install flash_attention
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


def build_emu3p5_vllm(
    model_path,
    tokenizer_path,
    vq_path,
    vq_type="ibq",
    vq_device="cuda:0",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    seed=6666,
    **kwargs,
):
    from vllm import LLM

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

    # resolution tokens
    resolution_map = {}
    resolution_str = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*"]
    for digit_str in resolution_str:
        resolution_map[tokenizer.encode(digit_str)[0]] = digit_str

    model = LLM(
        model_path,
        tokenizer=tokenizer_path,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=False,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_num_batched_tokens=26000,
        max_num_seqs=2,
        seed=seed,
        generation_config='vllm',
        scheduler_cls="vllm.v1.core.sched.batch_scheduler.Scheduler",
        compilation_config={
            "full_cuda_graph": True,
            "backend": "cudagraph",
            "cudagraph_capture_sizes": [1, 2],
        },
        additional_config={
            "boi_token_id": tokenizer.encode("<|image start|>")[0],
            "soi_token_id": tokenizer.encode("<|image token|>")[0],
            "eol_token_id": tokenizer.encode("<|extra_200|>")[0],
            "eoi_token_id": tokenizer.encode("<|image end|>")[0],
            "resolution_map": resolution_map,
        },
    )
    model.set_tokenizer(tokenizer)
    print(f"{model.llm_engine.vllm_config=}")

    return model, tokenizer, vq_model

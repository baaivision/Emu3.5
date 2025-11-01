# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from src.utils.logging_utils import setup_logger
cfg_name = Path(__file__).stem

model_path = "./weights/Emu3.5" # download from hf
vq_path = "./weights/Emu3.5-VisionTokenizer" # download from hf

tokenizer_path = "./src/tokenizer_emu3_ibq"
vq_type = "ibq"

# task_type in {"t2i", "x2i", "howto", "story", "explore", "vla"}
task_type = "howto"
# whether prompts include an input image token and provide reference_image paths
use_image = False

# saving config
exp_name = "emu3p5"
save_path = f"./outputs/{exp_name}/{task_type}"
save_to_proto = True
setup_logger(save_path)

hf_device = "auto"
vq_device = "cuda:0"
streaming = False
unconditional_type = "no_text"
classifier_free_guidance = 3.0 
max_new_tokens = 32768
image_area = 518400


def build_unc_and_template(task: str, with_image: bool):
    # System prompt header and role formatting remain consistent
    task_str = task.lower()
    if task_str == 'howto':
        extra_system_prompt = ' Please generate a response with interleaved text and images.'
    else:
        extra_system_prompt = ''

    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task.%s USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>"% (task_str, extra_system_prompt)
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task.%s USER: {question} ASSISTANT: <|extra_100|>" % (task_str, extra_system_prompt)
    return unc_p, tmpl

unc_prompt, template = build_unc_and_template(task_type, use_image)

sampling_params = dict(
    use_cache=True,
    # text token sampling config
    text_top_k=200,         
    text_top_p=0.8,         
    text_temperature=0.7,    

    # image token sampling config
    image_top_k=10240,      
    image_top_p=1.0,         
    image_temperature=1.0,  

    # general config
    top_k=131072,            # default topk (backward compatible)
    top_p=1.0,               # default top_p (backward compatible)
    temperature=1.0,         # default temperature (backward compatible)
    num_beams_per_group=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    max_new_tokens=max_new_tokens,
    guidance_scale=1.0,

    # enable differential sampling
    use_differential_sampling=True,
)

sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1
sampling_params["num_beams"] = sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]


special_tokens = dict(
    BOS="<|extra_203|>",
    EOS="<|extra_204|>",
    PAD="<|endoftext|>",
    EOL="<|extra_200|>",
    EOF="<|extra_201|>",
    TMS="<|extra_202|>",
    IMG="<|image token|>",
    BOI="<|image start|>",
    EOI="<|image end|>",
    BSS="<|extra_100|>",
    ESS="<|extra_101|>",
    BOG="<|extra_60|>",
    EOG="<|extra_61|>",
    BOC="<|extra_50|>",
    EOC="<|extra_51|>",
)

seed = 6666

# prompts config
# If use_image=True, each item should be a dict with {"prompt", "reference_image"}.
# If use_image=False, each item is a plain text string.

_prompts_base = [
    {
        "prompt": "How to cook Shrimp, Celery, and Pork Dumplings.",
        "reference_image": "",
    },

    # You can specify the length of the generated steps of Visual Guidance task by adding 'Please provide xxx steps for the task.'.
    # {
    #     "prompt": "How to cook Shrimp, Celery, and Pork Dumplings. Please provide 7 steps for the task.",
    #     "reference_image": "",
    # },
]

if use_image:
    prompts = _prompts_base
else:
    prompts = [p["prompt"] for p in _prompts_base]
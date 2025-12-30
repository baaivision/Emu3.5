# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from src.utils.logging_utils import setup_logger
cfg_name = Path(__file__).stem

model_path = "BAAI/Emu3.5-Image" # download from hf
vq_path = "BAAI/Emu3.5-VisionTokenizer" # download from hf

tokenizer_path = "./src/tokenizer_emu3_ibq"
vq_type = "ibq"

# task_type in {"t2i", "x2i", "howto", "story", "explore", "vla"}
task_type = "t2i"
# whether prompts include an input image token and provide reference_image paths
use_image = False

# saving config
exp_name = "emu3p5-image"
save_path = f"./outputs/{exp_name}/{task_type}"
save_to_proto = True
setup_logger(save_path)

hf_device = "auto"
vq_device = "cuda:0"
streaming = False
unconditional_type = "no_text"
classifier_free_guidance = 5.0 # For Emu3.5 model: we recommend set to 2
max_new_tokens = 5120
image_area = 1048576

aspect_ratios = {
    "4:3": "55*73",
    "21:9": "41*97",
    "16:9": "47*85",
    "3:2": "52*78",
    "1:1": "64*64",
    "3:4": "73*55",
    "9:16": "85*47",
    "2:3": "78*52",
    "default": "55*73",
    "auto": None,
}


def get_target_size(aspect_ratio: str):
    value = aspect_ratios.get(aspect_ratio, None)
    if value is None:
        return None, None

    h, w = map(int, value.split("*"))
    return h, w


# --- example usage ---
aspect_ratio = "default"     # User input, which can be replaced by "4:3", "1:1", "auto" etc.

target_height, target_width = get_target_size(aspect_ratio)

print(f"Aspect Ratio = {aspect_ratio}")
print(f"target_height = {target_height}, target_width = {target_width}")


def build_unc_and_template(task: str, with_image: bool):
    # System prompt header and role formatting remain consistent
    task_str = task.lower()
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>" % task_str
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: {question} ASSISTANT: <|extra_100|>" % task_str
    return unc_p, tmpl

unc_prompt, template = build_unc_and_template(task_type, use_image)

sampling_params = dict(
    use_cache=True,
    # text token sampling config
    text_top_k=1024,         
    text_top_p=0.9,         
    text_temperature=1.0,    

    # image token sampling config
    image_top_k=5120,      
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
        "prompt":"""A lively comic-style illustration depicting two humorous cartoon dogs interacting near a freshly dug backyard hole surrounded by scattered dirt, garden tools, blooming flowers, and a wooden fence background. At the upper-left side, Dog One stands nervously near the messy hole, ears down and eyes wide open with an expression of concern. Its speech bubble is an oval shape, outlined neatly with smooth, slightly rounded corners, positioned clearly above Dog One's head. Inside, clearly readable playful handwritten-style text emphasizes the dog's worried tone, saying, "You sure the humans won't notice this giant hole here?". Toward the lower-right side, Dog Two sits calmly and confidently with a cheerful, carefree expression, wagging its tail gently. Its speech bubble is rectangular with softly rounded edges, placed slightly overlapping with Dog One's speech bubble to guide the reader naturally downward diagonally across the frame. Dog Two's friendly, humorous response appears in a whimsical italicized comic font, clearly stating, "Relax! We'll just blame it on the neighbor's cat again!". Each speech bubble creats the playful and engaging backyard scene.""",
    },
]

if use_image:
    prompts = _prompts_base
else:
    prompts = [p["prompt"] for p in _prompts_base]

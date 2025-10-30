# -*- coding: utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from PIL import Image
import torch
import numpy as np


def smart_resize(image: Image.Image, area: int = 512 * 512, ds_factor: int = 16):
    width, height = image.size
    aspect_ratio = width / height
    new_height = int((area / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    # Round to nearest multiple of divisible_by
    new_height = ((new_height + ds_factor//2) // ds_factor) * ds_factor
    new_width = ((new_width + ds_factor//2) // ds_factor) * ds_factor
    return image.resize((new_width, new_height), Image.BICUBIC)


def format_image_string(tokenizer, image_tokens):
    image_string = ""
    h, w = image_tokens.shape
    for _h in range(h):
        row_string = ""
        for _w in range(w):
            row_string += "<|visual token {token_id:0>6d}|>".format(token_id=image_tokens[_h, _w])

        if _h < h - 1:
            row_string += tokenizer.eol_token
        image_string += row_string

    return "{image_start}{token_height}*{token_width}{image_token}{token_str}{image_end}".format(
        image_start=tokenizer.boi_token,
        token_height=h,
        token_width=w,
        image_token=tokenizer.img_token,
        token_str=image_string,
        image_end=tokenizer.eoi_token,
    )


@torch.no_grad()
def build_image(image, cfg, tokenizer, vq_model):
    image = smart_resize(image, cfg.image_area)
    w, h = image.size
    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype
    image = torch.tensor((np.array(image) / 127.5 - 1.0)).to(device, dtype).permute(2, 0, 1)
    _, _, token = vq_model.encode(image[None])
    token = token[-1].view(h // 16, w // 16)
    return format_image_string(tokenizer, token)

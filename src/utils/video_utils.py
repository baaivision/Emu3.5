# -*- coding: utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import imageio


def wrap_text(draw, text, font, max_width):
    lines = []
    current_line = ""
    
    i = 0
    while i < len(text):
        char = text[i]
        test_line = current_line + char
        
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= max_width:
            current_line = test_line
            i += 1
        else:
            if current_line:
                lines.append(current_line)
                current_line = ""
            else:
                current_line = char
                i += 1
    
    if current_line:
        lines.append(current_line)
    
    return lines


def plot_string(string, font_path="src/proto/assets/cangerjinkai.ttf", font_size=80, image_size=(512, 512), bg_color="white", text_color="black"):
    img = Image.new("RGB", image_size, color=bg_color)
    draw = ImageDraw.Draw(img)

    margin = 100
    max_width = max(image_size[0] - 2 * margin, 1)
    max_height = max(image_size[1] - 2 * margin, 1)

    def load_font(size):
        if font_path:
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                print(f"Failed to load font from {font_path}")
        return ImageFont.load_default()

    font = load_font(font_size)
    lines = wrap_text(draw, string, font, max_width)
    line_height = draw.textbbox((0, 0), "Ay", font=font)[3]
    total_text_height = line_height * max(len(lines), 1)

    if total_text_height > max_height:
        for size in range(font_size - 2, 9, -2):
            font = load_font(size)
            lines = wrap_text(draw, string, font, max_width)
            line_height = draw.textbbox((0, 0), "Ay", font=font)[3]
            total_text_height = line_height * max(len(lines), 1)
            if total_text_height <= max_height:
                break
        else:
            font = ImageFont.load_default()
            lines = wrap_text(draw, string, font, max_width)
            line_height = draw.textbbox((0, 0), "Ay", font=font)[3]
            total_text_height = line_height * max(len(lines), 1)

    y_offset = max(margin, (image_size[1] - total_text_height) // 2)

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x_offset = max(margin, (image_size[0] - text_width) // 2)
        draw.text((x_offset, y_offset), line, fill=text_color, font=font)
        y_offset += line_height

    return np.array(img)


def save_image_list_to_video(images, path, fps=1, quality='high'):
    os.makedirs(osp.dirname(path), exist_ok=True)
    
    if '.mp4' not in path and len(images) == 1:
        img = images[0]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy().astype(np.uint8)
        elif isinstance(img, Image.Image):
            img = np.array(img).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        Image.fromarray(img).save(path, quality=100)
        return
    
    func = lambda x: (
        x.detach().cpu().numpy().astype(np.uint8)
        if isinstance(x, torch.Tensor)
        else x.astype(np.uint8)
    )
    images = list(map(func, images))
    
    if quality == 'high':
        try:
            writer = imageio.get_writer(
                path,
                fps=fps,
                codec='libx264',
                ffmpeg_params=[
                    '-crf', '18',
                    '-preset', 'slow',
                    '-pix_fmt', 'yuv420p',
                ]
            )
            for image in images:
                writer.append_data(image)
            writer.close()
        except (TypeError, AttributeError):
            try:
                writer = imageio.get_writer(path, fps=fps, codec='libx264', macro_block_size=None)
                for image in images:
                    writer.append_data(image)
                writer.close()
            except Exception:
                with imageio.get_writer(path, fps=fps, mode='I') as writer:
                    for image in images:
                        writer.append_data(image)
    else:
        with imageio.get_writer(path, fps=fps, mode='I') as writer:
            for image in images:
                writer.append_data(image)

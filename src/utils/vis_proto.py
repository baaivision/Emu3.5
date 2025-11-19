# -*- coding: utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import os
import sys
from typing import Iterable, Tuple

import numpy as np
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.proto import emu_pb as story_pb
from src.utils.video_utils import plot_string, save_image_list_to_video


def parse_story(input_path: str) -> story_pb.Story:
    with open(input_path, 'rb') as f:
        story = story_pb.Story()
        story.ParseFromString(f.read())
        return story


def write_story_assets(story: story_pb.Story, output_path: str, generate_video: bool, fps: int) -> None:
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "000_question.txt"), 'w') as f:
        print(story.question, file=f)

    if story.summary and story.summary.strip():
        with open(os.path.join(output_path, "000_global_cot.txt"), 'w') as f:
            print(story.summary, file=f)

    idx = 1

    if len(story.reference_images) > 0:
        for i in range(len(story.reference_images)):
            with open(os.path.join(output_path, f"{i:03d}_reference_image.png"), 'wb') as f:
                f.write(story.reference_images[i].image.image_data)
        idx = len(story.reference_images)

    for clip in story.clips:
        for segment in clip.segments:
            with open(os.path.join(output_path, f"{idx:03d}_text.txt"), 'w') as f:
                print(segment.asr, file=f)
            for image_idx, image in enumerate(segment.images):
                with open(os.path.join(output_path, f"{idx:03d}_{image_idx:02d}_image.png"), 'wb') as f:
                    f.write(image.image.image_data)
                if image.chain_of_thought and image.chain_of_thought.strip():
                    with open(os.path.join(output_path, f"{idx:03d}_{image_idx:02d}_image_cot.txt"), 'w') as f:
                        print(image.chain_of_thought, file=f)
            idx += 1

    if generate_video:
        export_story_video(story, output_path, fps)


def export_story_video(story: story_pb.Story, output_path: str, fps: int) -> None:
    video_images = []
    target_size = None

    for ref_img_data in story.reference_images:
        img = Image.open(io.BytesIO(ref_img_data.image.image_data))
        img = img.convert('RGB')
        if target_size is None:
            target_size = img.size

    for clip in story.clips:
        for segment in clip.segments:
            for image in segment.images:
                img = Image.open(io.BytesIO(image.image.image_data))
                img = img.convert('RGB')
                if target_size is None:
                    target_size = img.size

    if target_size is None:
        target_size = (512, 512)

    if story.question and story.question.strip():
        question_img = plot_string(story.question, image_size=(target_size[0], target_size[1]))
        video_images.append(question_img)

    for img_array in story.reference_images:
        img = Image.open(io.BytesIO(img_array.image.image_data))
        img = img.convert('RGB')
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        video_images.append(np.array(img))

    for clip in story.clips:
        for segment in clip.segments:
            if segment.asr and segment.asr.strip():
                asr_img = plot_string(segment.asr, image_size=(target_size[0], target_size[1]))
                video_images.append(asr_img)

            for image in segment.images:
                img = Image.open(io.BytesIO(image.image.image_data))
                img = img.convert('RGB')
                if img.size != target_size:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                video_images.append(np.array(img))

    if video_images:
        video_path = os.path.join(output_path, "video.mp4")
        save_image_list_to_video(video_images, video_path, fps=fps, quality='high')
        print(f"Video saved to: {video_path}")


def iter_proto_files(input_dir: str) -> Iterable[Tuple[str, str]]:
    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.lower().endswith(('.pb', '.proto', '.bin')):
                full_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(full_path, input_dir)
                yield full_path, rel_path


def process_single_file(input_path: str, output_path: str, generate_video: bool, fps: int) -> None:
    story = parse_story(input_path)
    write_story_assets(story, output_path, generate_video, fps)
    print(f"Visualization saved to: {output_path}")


def process_directory(input_dir: str, output_root: str, generate_video: bool, fps: int) -> None:
    has_processed = False
    for abs_proto_path, rel_proto_path in iter_proto_files(input_dir):
        rel_without_ext, _ = os.path.splitext(rel_proto_path)
        target_dir = os.path.join(output_root, rel_without_ext)
        process_single_file(abs_proto_path, target_dir, generate_video, fps)
        has_processed = True

    if not has_processed:
        print(f"No protobuf files found under: {input_dir}")


def default_output_for_file(input_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(input_path))
    file_stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(base_dir, "results", file_stem)


def default_output_for_directory(input_dir: str) -> str:
    parent_dir = os.path.dirname(os.path.abspath(os.path.normpath(input_dir)))
    return os.path.join(parent_dir, "results")


def main():
    parser = argparse.ArgumentParser(description='Visualize protobuf story files')
    parser.add_argument('--input', '-i', required=True, help='Input protobuf file or directory path')
    parser.add_argument('--output', '-o', help='Output directory path (optional)')
    parser.add_argument('--video', action='store_true', help='Generate video from protobuf content')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second for video (default: 1)')
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output) if args.output else None

    if os.path.isdir(input_path):
        output_root = output_path if output_path else default_output_for_directory(input_path)
        process_directory(input_path, output_root, args.video, args.fps)
    elif os.path.isfile(input_path):
        target_output = output_path if output_path else default_output_for_file(input_path)
        process_single_file(input_path, target_output, args.video, args.fps)
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()

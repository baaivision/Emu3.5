# -*- coding: utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
import numpy as np
from PIL import Image
import io

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.proto import emu_pb as story_pb
from src.utils.video_utils import plot_string, save_image_list_to_video

def main():
    parser = argparse.ArgumentParser(description='Visualize protobuf story files')
    parser.add_argument('--input', '-i', required=True, help='Input protobuf file path')
    parser.add_argument('--output', '-o', required=True, help='Output directory path')
    parser.add_argument('--video', action='store_true', help='Generate video from protobuf content')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second for video (default: 1)')
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    os.makedirs(output_path, exist_ok=True)
    
    with open(input_path, 'rb') as f:
        story = story_pb.Story()
        story.ParseFromString(f.read())

    with open(f"{output_path}/000_question.txt", 'w') as f:
        print(story.question, file=f)

    if story.summary and story.summary.strip():
        with open(f"{output_path}/000_global_cot.txt", 'w') as f:
            print(story.summary, file=f)

    idx = 1
    
    if len(story.reference_images) > 0:
        for i in range(len(story.reference_images)):
            with open(f"{output_path}/{i:03d}_reference_image.png", 'wb') as f:
                f.write(story.reference_images[i].image.image_data)
        idx = len(story.reference_images)
        
    for c in story.clips:
        for s in c.segments:
            with open(f"{output_path}/{idx:03d}_text.txt", 'w') as f:
                print(s.asr, file=f)
            for im_idx, im in enumerate(s.images):
                with open(f"{output_path}/{idx:03d}_{im_idx:02d}_image.png", 'wb') as f:
                    f.write(im.image.image_data)
                if im.chain_of_thought and im.chain_of_thought.strip():
                    with open(f"{output_path}/{idx:03d}_{im_idx:02d}_image_cot.txt", 'w') as f:
                        print(im.chain_of_thought, file=f)
            idx += 1
    
    if args.video:
        video_images = []
        target_size = None
        
        for ref_img_data in story.reference_images:
            img = Image.open(io.BytesIO(ref_img_data.image.image_data))
            img = img.convert('RGB')
            if target_size is None:
                target_size = img.size
        
        for c in story.clips:
            for s in c.segments:
                for im in s.images:
                    img = Image.open(io.BytesIO(im.image.image_data))
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
        
        for c in story.clips:
            for s in c.segments:
                if s.asr and s.asr.strip():
                    asr_img = plot_string(s.asr, image_size=(target_size[0], target_size[1]))
                    video_images.append(asr_img)
                
                for im in s.images:
                    img = Image.open(io.BytesIO(im.image.image_data))
                    img = img.convert('RGB')
                    if img.size != target_size:
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                    video_images.append(np.array(img))
        
        if video_images:
            video_path = f"{output_path}/video.mp4"
            save_image_list_to_video(video_images, video_path, fps=args.fps, quality='high')
            print(f"Video saved to: {video_path}")

if __name__ == "__main__":
    main()

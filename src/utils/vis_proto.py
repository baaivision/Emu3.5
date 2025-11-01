# -*- coding: utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
# Ensure project root is on sys.path so `src` is importable when run as a script
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.proto import emu_pb as story_pb

def main():
    parser = argparse.ArgumentParser(description='Visualize protobuf story files')
    parser.add_argument('--input', '-i', required=True, help='Input protobuf file path')
    parser.add_argument('--output', '-o', required=True, help='Output directory path')
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    os.makedirs(output_path, exist_ok=True)
    
    with open(input_path, 'rb') as f:
        story = story_pb.Story()
        story.ParseFromString(f.read())

    with open(f"{output_path}/000_question.txt", 'w') as f:
        print(story.question, file=f)

    if len(story.reference_images) > 0:
        if len(story.reference_images) == 1:
            with open(f"{output_path}/000_reference_image.png", 'wb') as f:
                f.write(story.reference_images[0].image.image_data)
        else:
            for i in range(len(story.reference_images)):
                with open(f"{output_path}/000_reference_image_{i}.png", 'wb') as f:
                    f.write(story.reference_images[i].image.image_data)

    idx = 1
    for c in story.clips:
        for s in c.segments:
            with open(f"{output_path}/{idx:03d}_prompt.txt", 'w') as f:
                print(s.asr, file=f)
            for im_idx, im in enumerate(s.images):
                with open(f"{output_path}/{idx:03d}_{im_idx:02d}_image.png", 'wb') as f:
                    f.write(im.image.image_data)
            idx += 1

if __name__ == "__main__":
    main()

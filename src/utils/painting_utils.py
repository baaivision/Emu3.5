# -*- coding: utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import re
import os.path as osp

from PIL import Image

from src.proto import emu_pb as story_pb

class ProtoWriter:
    
    def __init__(self):
        self.story = story_pb.Story()
        self.image_tensor = None

    def clear(self):
        self.story = story_pb.Story()
        self.image_tensor = None

    def extend(self, multimodal_output):
        for t, c in multimodal_output:
            match t:
                case "question":
                    self.story.question = c
                case "global_cot":
                    self.story.summary = c
                case "image_cot":
                    image = story_pb.ImageMeta()
                    image.chain_of_thought = c
                    self._put_last_image(image)
                case "text":
                    self._put_last_clip(self._build_clip(c))
                case "image":
                    image = self._get_last_image()
                    image.image.CopyFrom(self._build_image(c))
                    self._put_last_image(image)
                case "reference_image":
                    image = story_pb.ImageMeta()
                    image.image.CopyFrom(self._build_image(c))
                    self.story.reference_images.append(image)
                case _:
                    raise NotImplementedError(f"Unsupported data type {t}")

    def save(self, path):
        self._check_last_image()
        with open(path, 'wb') as f:
            f.write(self.story.SerializeToString())


    def _build_clip(self, text_content=""):
        clip = story_pb.Clip()
        clip.clip_id = f"clip_{len(self.story.clips):04d}"
        segment = story_pb.Segment()
        segment.asr = text_content

        clip.segments.append(segment)
        return clip

    def _build_image(self, image):
        im = story_pb.Image()
        im.width, im.height = image.size
        im.format = story_pb.ImageFormat.PNG

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        im.image_data = img_byte_arr.getvalue()

        return im

    def _get_last_image(self):
        if not self.story.clips:
            self._put_last_clip(self._build_clip())

        if self.story.clips[-1].segments[0].images and not self.story.clips[-1].segments[0].images[-1].image.image_data:
            image = self.story.clips[-1].segments[0].images[-1]
            del self.story.clips[-1].segments[0].images[-1]
        else:
            image = story_pb.ImageMeta()

        return image

    def _put_last_image(self, image):
        if not self.story.clips:
            self._put_last_clip(self._build_clip())

        self.story.clips[-1].segments[0].images.append(image)

    def _put_last_clip(self, clip):
        self.story.clips.append(clip)

    def _check_last_image(self):
        image = self._get_last_image()
        if image.image.image_data:
            self._put_last_image(image)

# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F
from transformers.generation import LogitsProcessor

BOS = 151849
EOS = 151850
IMG = 151851
BOI = 151852
EOI = 151853
EOL = 151846
EOF = 151847
BOV = 151854


class UnbatchedClassifierFreeGuidanceLogitsForVisualTokenProcessor(LogitsProcessor):

    def __init__(
        self,
        guidance_scale: float,
        model,
        tokenizer,
        unconditional_ids: torch.LongTensor,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        full_unconditional_ids: Optional[torch.LongTensor] = None,
        full_unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        allowed_tokens_control: bool = True,
        force_same_image_size: bool = True,
        unconditional_type: str = "no_text", # options: no_text, no_prev_text, no_prev_modal, no_text_img_cfg, etc.
        target_height: Optional[int] = None,  # added parameter is used to specify the target height
        target_width: Optional[int] = None,   # added parameter is used to specify the target width
        image_cfg_scale: float = 1.0,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.tokenizer = tokenizer

        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": (
                unconditional_attention_mask
                if unconditional_attention_mask is not None
                else torch.ones_like(unconditional_ids, dtype=torch.long)
            ),
            "default_input_ids": unconditional_ids,
            "default_attention_mask": (
                unconditional_attention_mask
                if unconditional_attention_mask is not None
                else torch.ones_like(unconditional_ids, dtype=torch.long)
            ),
            "first_pass": True,
            "past_key_values": None,
        }

        self.image_cfg_scale = image_cfg_scale
        # heuristic combination example: fuc + a*(uc - fuc) + b*(logit - un)
        self.full_unconditional_context = {
            "input_ids": full_unconditional_ids if full_unconditional_ids is not None else unconditional_ids,
            "attention_mask": (
                unconditional_attention_mask
                if unconditional_attention_mask is not None
                else torch.ones_like(unconditional_ids, dtype=torch.long)
            ),
            "default_input_ids": full_unconditional_ids if full_unconditional_ids is not None else unconditional_ids,
            "default_attention_mask": (
                full_unconditional_attention_mask
                if full_unconditional_attention_mask is not None
                else (
                    unconditional_attention_mask
                    if unconditional_attention_mask is not None
                    else
                    torch.ones_like(unconditional_ids, dtype=torch.long)
                )
            ),
            "first_pass": True,
            "past_key_values": None,
        }

        self.use_cache = use_cache

        self.first_in_image = True
        self.in_image = False
        self.in_visual = False
        self.image_nums = 0

        self.allowed_tokens_control = allowed_tokens_control

        self.height = target_height 
        self.width = target_width
        if self.height is not None and self.width is not None:
            print(f"[INFO] User defined: height: {self.height}, width: {self.width}")
        else:
            print(f"[INFO] Auto height, width")
        self.hw_tokens = None
        self.force_same_image_size = force_same_image_size
        self.unconditional_type = unconditional_type

        self.text_segment = 0

        if IMG in unconditional_ids[0]:
            self.parse_hw(unconditional_ids)
            self.first_in_image = False

    # used only for visual tokens with classifier-free guidance
    def get_unconditional_logits(self, input_ids):
        # update cache
        input_ids, attention_mask = self.update_unconditional_context(input_ids, target="unconditional")
        if self.use_cache and not self.unconditional_context["first_pass"]:
            input_ids = input_ids[:, -1:]
        else:
            self.unconditional_context["first_pass"] = False

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.use_cache,
            past_key_values=self.unconditional_context["past_key_values"],
        )

        if self.use_cache:
            self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        if self.unconditional_type == "no_text_img_cfg":
            input_ids, attention_mask = self.update_unconditional_context(input_ids, target="full_unconditional")

            if self.use_cache and not self.full_unconditional_context["first_pass"]:
                input_ids = input_ids[:, -1:]
            else:
                self.full_unconditional_context["first_pass"] = False

            func_out = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=self.use_cache,
                past_key_values=self.full_unconditional_context["past_key_values"],
            )

            if self.use_cache:
                self.full_unconditional_context["past_key_values"] = out.get("past_key_values", None)

            return out.logits[:, -1], func_out.logits[:, -1]

        return out.logits[:, -1]

    def __call__(self, input_ids, scores):
        # IMAGE MODE
        if input_ids[0][-1] == BOI:
            self.set_unconditional_context(input_ids)
        # TEXT MODE
        if input_ids[0][-1] == EOI:
            self.exit_image(input_ids)

        if input_ids[0][-1] != BOI and input_ids[0][-2] == EOI:
            self.text_segment += 1

        # all image tokens between BOI and EOI are handled by in_image_logits_processor
        N = input_ids.shape[1]
        if self.in_image:
            scores = self.in_image_logits_processor(input_ids, scores)
        else:
            scores = self.in_text_logits_processor(input_ids, scores)

        return scores

    def in_image_logits_processor(self, input_ids, scores):
        # all visual-related logits must call get_unconditional_logits to keep unconditional cache updated
        unc_scores = self.get_unconditional_logits(input_ids)

        if self.in_visual:
            # generating visual tokens
            img_idx = self.find_last_token_index(input_ids[0], IMG)
            vis_idx = input_ids.shape[1] - img_idx
            # eoi
            if vis_idx == self.height * (self.width + 1):
                mask = torch.full_like(scores, -math.inf)
                mask[:, EOI] = 0
                scores = scores + mask
                return scores
            # eol
            elif vis_idx % (self.width + 1) == 0:
                mask = torch.full_like(scores, -math.inf)
                mask[:, EOL] = 0
                scores = scores + mask
                return scores
            # visual token
            else:
                scores = self.apply_cfg(scores, unc_scores, self.guidance_scale, self.image_cfg_scale)
                mask = torch.full_like(scores, -math.inf)
                mask[:, BOV:] = 0
                scores = scores + mask
                return scores
        else:
            # height and width have been generated; confirm h and w
            if input_ids[0][-1] == IMG:
                self.in_visual = True
                self.image_nums += 1
                print(f"[INFO] Generating image#{self.image_nums}...")
                if self.first_in_image or not self.force_same_image_size:
                    if self.height is None or self.width is None:  # Only when the size is not specified will the analysis be conducted.
                        self.parse_hw(input_ids)
                    self.first_in_image = False
                
                # the first visual token
                scores = self.apply_cfg(scores, unc_scores, self.guidance_scale, self.image_cfg_scale)

                # mask non visual tokens
                mask = torch.full_like(scores, -math.inf)
                mask[:, BOV:] = 0
                scores = scores + mask
                return scores
            # not yet at IMG token: we are generating h x w tokens
            else:
                if self.height is not None and self.width is not None:
                    # If the size is specified, the corresponding h and w tokens will be forcibly generated.
                    boi_idx = self.find_last_token_index(input_ids[0], BOI)
                    hw_idx = input_ids.shape[1] - boi_idx
                    hw_tokens = self.tokenizer.encode(f"{str(self.height)}*{str(self.width)}", add_special_tokens=False)

                    if hw_idx <= len(hw_tokens):  # height part
                        allowed_tokens = [hw_tokens[hw_idx-1]]
                    else:  # IMG token
                        allowed_tokens = [IMG]
                        
                    mask = torch.full_like(scores, -math.inf)
                    for token in allowed_tokens:
                        mask[:, token] = 0
                    scores = scores + mask
                    return scores
                else:
                    if self.first_in_image or not self.force_same_image_size:
                        # force output format for h x w sequence
                        boi_idx = self.find_last_token_index(input_ids[0], BOI)
                        hw_idx = input_ids.shape[1] - boi_idx
                        mask = torch.full_like(scores, -math.inf)
                        if hw_idx > 5:
                            mask[:, IMG] = 0
                        elif hw_idx in (1, 4):
                            # example mapping: '1' -> token 16, '9' -> token 24
                            mask[:, 16:25] = 0
                        elif hw_idx == 3:
                            # '*' -> token 9
                            mask[:, 9] = 0
                        elif hw_idx in (2, 5):
                            # '0' -> token 15, '9' -> token 24
                            mask[:, 15:25] = 0
                        scores = scores + mask
                        return scores
                    else:
                        # force same image size: must reproduce previous h and w tokens
                        boi_idx = self.find_last_token_index(input_ids[0], BOI)
                        hw_idx = input_ids.shape[1] - boi_idx
                        mask = torch.full_like(scores, -math.inf)
                        if hw_idx > len(self.hw_tokens):
                            mask[:, IMG] = 0
                        else:
                            mask[:, self.hw_tokens[hw_idx - 1]] = 0
                        scores = scores + mask
                        return scores

    def in_text_logits_processor(self, input_ids, scores):
        # for text tokens, do not apply CFG here
        mask = torch.full_like(scores, -math.inf)
        mask[:, :BOV] = 0
        scores = scores + mask
        return scores

    def parse_hw(self, input_ids):
        if self.height is not None and self.width is not None and self.force_same_image_size:
            return

        # Find indices of last BOI and IMG tokens in the sequence
        seq = input_ids[0]
        img_indices = (seq == IMG).nonzero().flatten()
        boi_indices = (seq == BOI).nonzero().flatten()

        # Get the last occurrence of each token
        last_img_pos = img_indices[-1].item()
        last_boi_pos = boi_indices[-1].item()

        # Get tokens between last BOI and IMG
        self.hw_tokens = seq[last_boi_pos+1:last_img_pos]

        # Decode tokens to string and parse dimensions
        hw_str = self.tokenizer.decode(self.hw_tokens)
        h, w = hw_str.split('*')
        self.height = int(h)
        self.width = int(w)
        print(f"[INFO] hw_str: {hw_str}, H_px: {self.height * 16}, W_px: {self.width * 16}")

    def find_last_token_index(self, seq, token_id):
        # seq shape: (N,)
        token_indices = (seq == token_id).nonzero().flatten()
        if len(token_indices) == 0:
            return -1
        return token_indices[-1].item()

    def find_first_token_index(self, seq, token_id):
        token_indices = (seq == token_id).nonzero().flatten()
        if len(token_indices) == 0:
            return -1
        return token_indices[0].item()

    def apply_cfg(self, scores, unc_scores, cfg_scale, img_cfg_scale):
        match self.unconditional_type:
            case "no_text":
                # applying no_text CFG
                scores = F.log_softmax(scores, dim=-1)
                unc_scores = F.log_softmax(unc_scores, dim=-1)
                # c = unc + cfg * (c - unc)
                scores = cfg_scale * (scores - unc_scores) + unc_scores
                return scores
            case "no_text_img_cfg":
                # applying no_text and no_any CFG
                no_text_scores, no_any_scores = unc_scores
                scores = F.log_softmax(scores, dim=-1)
                no_text_scores = F.log_softmax(no_text_scores, dim=-1)
                no_any_scores = F.log_softmax(no_any_scores, dim=-1)
                # c = unc + cfg_1 * (unc_im - unc) + cfg_2 * (c - unc_im)
                scores = no_any_scores + img_cfg_scale * (no_text_scores - no_any_scores) + cfg_scale * (scores - no_text_scores)
                return scores
            case _:
                raise NotImplementedError(f"Unsupported unconditional type {self.unconditional_type}")

    def set_unconditional_context(self, input_ids):
        self.in_image = True

        if self.unconditional_type == "no_text" or self.unconditional_type == "uncondition_tokens":
            # no special handling
            return
        elif self.unconditional_type == "no_text_img_cfg":
            self.full_unconditional_context["input_ids"] = self.full_unconditional_context["default_input_ids"]
            self.full_unconditional_context["attention_mask"] = self.full_unconditional_context["default_attention_mask"]
            self.full_unconditional_context["first_pass"] = True
            self.full_unconditional_context["past_key_values"] = None
        else:
            raise ValueError(f"Unconditional type {self.unconditional_type} not supported")

    def update_unconditional_context(self, input_ids, target="unconditional"):
        if target == "unconditional":
            self.unconditional_context["input_ids"] = torch.cat(
                [
                    self.unconditional_context["input_ids"],
                    input_ids[:, -1:],
                ],
                dim=1,
            )
            self.unconditional_context["attention_mask"] = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            return self.unconditional_context["input_ids"], self.unconditional_context["attention_mask"]
        elif target == "full_unconditional":
            self.full_unconditional_context["input_ids"] = torch.cat(
                [
                    self.full_unconditional_context["input_ids"],
                    input_ids[:, -1:],
                ],
                dim=1,
            )
            self.full_unconditional_context["attention_mask"] = torch.cat(
                [
                    self.full_unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            return self.full_unconditional_context["input_ids"], self.full_unconditional_context["attention_mask"]


    def exit_image(self, input_ids):
        self.in_image = False
        self.in_visual = False
        # update eoi to unconditional cache
        self.get_unconditional_logits(input_ids)


class UnbatchedClassifierFreeGuidanceLogitsForVisualTokenWithDifferentialTopKProcessor(UnbatchedClassifierFreeGuidanceLogitsForVisualTokenProcessor):
    """
    Extend the original CFG processor and add the differentiated TopK function.
    Use different topk, top_p, and temperature parameters when generating text tokens and image tokens.
    """

    def __init__(
        self,
        guidance_scale: float,
        model,
        tokenizer,
        unconditional_ids: Optional[torch.LongTensor],
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        full_unconditional_ids: Optional[torch.LongTensor] = None,
        full_unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        allowed_tokens_control: bool = True,
        force_same_image_size: bool = True,
        unconditional_type: str = "no_text",
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        image_cfg_scale: float = 1.0,
        # added differentiated topk parameter
        use_differential_sampling: bool = False,
        text_top_k: int = 1024,
        image_top_k: int = 10240,
        text_top_p: float = 0.9,
        image_top_p: float = 1.0,
        text_temperature: float = 1.0,
        image_temperature: float = 1.0,
        **kwargs,
    ):

        super().__init__(
            guidance_scale=guidance_scale,
            model=model,
            tokenizer=tokenizer,
            unconditional_ids=unconditional_ids,
            unconditional_attention_mask=unconditional_attention_mask,
            full_unconditional_ids=full_unconditional_ids,
            full_unconditional_attention_mask=full_unconditional_attention_mask,
            use_cache=use_cache,
            allowed_tokens_control=allowed_tokens_control,
            force_same_image_size=force_same_image_size,
            unconditional_type=unconditional_type,
            target_height=target_height,
            target_width=target_width,
            **kwargs,
        )

        # differential sampling parameters
        self.use_differential_sampling = use_differential_sampling
        self.text_top_k = text_top_k
        self.image_top_k = image_top_k
        self.text_top_p = text_top_p
        self.image_top_p = image_top_p
        self.text_temperature = text_temperature
        self.image_temperature = image_temperature

    def apply_differential_topk(self, scores, is_image_generation=False):
        """Apply mode-specific top_k/top_p/temperature depending on image/text generation."""
        if not self.use_differential_sampling:
            return scores

        # choose params based on current mode
        if is_image_generation:
            current_top_k = self.image_top_k
            current_top_p = self.image_top_p
            current_temperature = self.image_temperature
            mode = "IMAGE"
        else:
            current_top_k = self.text_top_k
            current_top_p = self.text_top_p
            current_temperature = self.text_temperature
            mode = "TEXT"

        # temperature scaling
        if current_temperature != 1.0:
            scores = scores / current_temperature

        # Top-K filtering
        if current_top_k > 0 and current_top_k < scores.size(-1):
            top_k = min(current_top_k, scores.size(-1))
            # mask everything below kth logit
            indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
            scores[indices_to_remove] = float('-inf')

        # Top-P (nucleus) filtering
        if current_top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scores, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # find positions exceeding cumulative probability threshold
            sorted_indices_to_remove = cumulative_probs > current_top_p
            # keep first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # map mask back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores[indices_to_remove] = float('-inf')

        return scores

    def in_image_logits_processor(self, input_ids, scores):
        """Override: add differential top-k for image tokens."""
        # apply base processing first
        scores = super().in_image_logits_processor(input_ids, scores)

        # then apply differential top-k for image tokens
        scores = self.apply_differential_topk(scores, is_image_generation=True)

        return scores

    def in_text_logits_processor(self, input_ids, scores):
        """Override: add differential top-k for text tokens."""
        # apply base processing first
        scores = super().in_text_logits_processor(input_ids, scores)

        # then apply differential top-k for text tokens
        scores = self.apply_differential_topk(scores, is_image_generation=False)

        return scores
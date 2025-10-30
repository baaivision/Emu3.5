<div align='center'>
<h1>Emu3.5: Native Multimodal Models are World Learners</h1>

Emu3.5 Team, BAAI

[Project Page](https://emu.world/) | [🤗HF Models](https://huggingface.co/collections/BAAI/emu35) | [Paper](https://emu.world/Emu35_tech_report.pdf)
</div>

<div align='center'>
<img src="./assets/arch.png" class="interpolation-image" alt="arch." height="100%" width="100%" />
</div>

We introduce Emu3.5, a large-scale multimodal world model that natively predicts the next state across vision and language. Emu3.5 is pre-trained end-to-end with a unified next-token prediction objective on a corpus of vision-language interleaved data containing over 10 trillion tokens, primarily derived from sequential frames and transcripts of internet videos. The model naturally accepts interleaved vision-language inputs and generates interleaved vision-language outputs. Emu3.5 is further post-trained with large-scale reinforcement learning to enhance multi-modal reasoning and generation. To improve inference efficiency, we propose Discrete Diffusion Adaptation (DiDA), which converts token-by-token decoding into bidirectional parallel prediction, accelerating per-image inference by about 20×without sacrificing performance. Emu3.5 exhibits strong native multimodal capabilities, including long-horizon vision-language generation, any-to-image (X2I) generation, and complex text-rich image generation. It also exhibits generalizable world-modeling abilities, enabling spatiotemporally consistent world exploration and open-world embodied manipulation across diverse scenarios and tasks. For comparison, Emu3.5 achieves performance comparable to Gemini 2.5 Flash Image (Nano Banana) on image generation and editing tasks and demonstrates superior results on a suite of interleaved generation tasks.

<div align='center'>
<img src="./assets/co.png" class="interpolation-image" alt="arch." height="90%" width="90%" />
</div>

## Table of Contents

1. [Model & Weights](#1-model--weights)
2. [Quick Start](#2-quick-start)
3. [Schedule](#3-schedule)
4. [Citation](#4-citation)

## 1. Model & Weights

| Model name               | HF Weight |
| ------------------------ | --------- |
| Emu3.5               | [🤗 HF link](https://huggingface.co/BAAI/Emu3.5/tree/main) |
| Emu3.5-Image                | [🤗 HF link](https://huggingface.co/BAAI/Emu3.5-Image/tree/main) |
| Emu3.5-VisionTokenizer     | [🤗 HF link](https://huggingface.co/BAAI/Emu3.5-VisionTokenizer/tree/main) |

## 2. Quick Start

### Environment Setup

```bash
git clone https://github.com/baaivision/Emu3.5
cd Emu3.5
pip install -r requirements.txt
pip install flash_attn==2.8.3 --no-build-isolation
```
### Configuration

Edit `configs/config.py` to set:

- Paths: `model_path`, `vq_path`
- Task template: `task_type in {t2i, x2i, howto, story, explore, vla}`, `use_image` controls `<|IMAGE|>` usage (set to true when reference images are provided)
- Sampling: `sampling_params` (classifier_free_guidance, temperature, top_k/top_p, etc.)

### Run Inference

```bash
python inference.py --cfg configs/config.py
```

Protobuf outputs are written to `outputs/<exp_name>/proto/`. For better throughput, we recommend ≥2 GPUs.

### Visualize Protobuf Outputs

To visualize generated protobuf files:

```bash
python src/utils/vis_proto.py --input <input_proto_file> --output <output_dir>
```

## 3. Schedule

- [x] Inference Code
- [ ] Advanced Image Decoder
- [ ] Discrete Diffusion Adaptation(DiDA)


## 4. Citation

<!-- ```bibtex
@misc{emu352025,
  title  = {Emu3.5},
  year   = {2025},
}
``` -->


<div align='center'>
<h1>Emu3.5: Native Multimodal Models are World Learners</h1>

Emu3.5 Team, BAAI

[Project Page](https://emu.world/) | [🤗HF Models](https://huggingface.co/collections/BAAI/emu35) | [Paper](https://emu.world/Emu35_tech_report.pdf)
</div>

<div align='center'>
<img src="./assets/arch.png" class="interpolation-image" alt="arch." height="100%" width="100%" />
</div>

🌍 Emu3.5 Highlights

🧠 Unified World Modeling
Natively predicts the next state across vision and language, enabling consistent world understanding and generation.

🧩 End-to-End Pretraining Objective
Trained with a unified next-token prediction over interleaved vision-language data, jointly modeling perception and semantics.

📚 10-Trillion-Token Corpus
Pre-trained on 10T+ vision-language tokens from sequential internet video frames and transcripts, forming rich spatiotemporal grounding.

🔄 Native Multimodal I/O
Accepts and generates interleaved visual-text sequences without task-specific adapters or modality segregation.

🎯 Reinforcement Learning Post-Training
Large-scale RL fine-tuning enhances multimodal reasoning, compositionality, and generative fidelity.

⚡ Discrete Diffusion Adaptation (DiDA)
Transforms token-by-token decoding into bidirectional parallel prediction, achieving ≈20× faster per-image inference with no quality drop.

🖼️ Versatile Generation Abilities
Supports long-horizon vision-language generation, any-to-image (X2I) synthesis, and text-rich image creation.

🌐 Generalizable World Modeling
Demonstrates spatiotemporally consistent exploration and open-world embodied manipulation across diverse scenarios.

🏆 Performance & Comparison
Matches Gemini 2.5 Flash Image (Nano Banana) in image generation/editing and outperforms on interleaved generation tasks.

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


<div align='center'>
<h1>Emu3.5: Native Multimodal Models are World Learners</h1>

Emu3.5 Team, BAAI

[Project Page](https://emu.world/) | [🤗HF Models](https://huggingface.co/collections/BAAI/emu35) | [Paper](https://arxiv.org/pdf/2510.26583)
</div>


<div align='center'>
<img src="./assets/arch.png" class="interpolation-image" alt="arch." height="100%" width="100%" />
</div>


<div align='center'>
<img src="./assets/co.png" class="interpolation-image" alt="arch." height="90%" width="90%" />
</div>


|  🔹 | **Core Concept**                         | **Description**                                                                                                                            |
| :-: | :--------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
|  🧠 | **Unified World Modeling**               | Predicts the **next state jointly across vision and language**, enabling coherent **world modeling** and **generation**.              |
|  🧩 | **End-to-End Pretraining**               | Trained with a **unified next-token prediction** objective over **interleaved vision–language sequences**.                                 |
|  📚 | **Over 10T+ Multimodal Tokens**               | Pre-trained on **over 10 trillion interleaved tokens** from **video frames** and **transcripts**, capturing **spatiotemporal structure**.       |
|  🔄 | **Native Multimodal I/O**                | Processes and generates **interleaved visual–text sequences** without **modality adapters** or **task-specific heads**.                    |
|  🎯 | **RL Post-Training**                     | Large-scale **reinforcement learning** enhances **reasoning**, **compositionality**, and **generation quality**.                           |
|  ⚡  | **Discrete Diffusion Adaptation (DiDA)** | Converts **sequential decoding → bidirectional parallel prediction**, achieving **≈20× faster inference without performance loss**.      |
| 🖼️ | **Versatile Generation**                 | Excels in **long-horizon vision–language generation**, **any-to-image (X2I)** synthesis, and **text-rich image creation**.                 |
|  🌐 | **Generalizable World Modeling**         | Enables **spatiotemporally consistent world exploration**, and **open-world embodied manipulation** across diverse scenarios.          |
|  🏆 | **Performance Benchmark**                | Matches **Gemini 2.5 Flash Image (Nano Banana)** on **image generation/editing**, and **outperforms** on **interleaved generation tasks**. |



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


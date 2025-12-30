<div align='center'>
<h1>Emu3.5: Native Multimodal Models are World Learners</h1>

Emu3.5 Team, BAAI

[Project Page](https://emu.world/pages/web/landingPage) | [🤗HF Models](https://huggingface.co/collections/BAAI/emu35) | [Paper](https://arxiv.org/pdf/2510.26583) | [App](https://emu.world/pages/web/home?route=index)
</div>


> 🔔 **Latest**: Emu3.5 Web & Mobile Apps and vLLM offline inference are live — see [🔥 News](#news) for details.


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


<a id="news"></a>

## 🔥 News

- **2025-11-28 · 🌐 Emu3.5 Web & Mobile Apps Live** — Official product experience is **now available** on the web at [zh.emu.world](https://zh.emu.world) (Mainland China) and [emu.world](https://emu.world) (global) 🎉 The new homepage highlights featured cases and a “Get Started” entry, while the workspace and mobile apps bring together creation, inspiration feed, history, profile, and language switch across web, Android APK, and H5. *([See more details](#official-web--mobile-apps) below.)*
- **2025-11-19 · 🚀 vLLM Offline Inference Released** — Meet `inference_vllm.py` with a new cond/uncond batch scheduler, delivering **4–5× faster end-to-end generation** on vLLM 0.11.0 across Emu3.5 tasks. Jump to [#Run Inference with vLLM](#run-inference-with-vllm) for setup guidance and see PR [#47](https://github.com/baaivision/Emu3.5/pull/47) for full details.
- **2025-11-17 · 🎛️ Gradio Demo (Transformers Backend)** — Introduced `gradio_demo_image.py` and `gradio_demo_interleave.py` presets for the standard Transformers runtime, providing turnkey T2I/X2I and interleaved generation experiences with streaming output. Try the commands in [#Gradio Demo](#3-gradio-demo) to launch both UIs locally.

## Table of Contents

1. [Model & Weights](#1-model--weights)
2. [Quick Start](#2-quick-start)
3. [Gradio Demo](#3-gradio-demo)
4. [Schedule](#4-schedule)
5. [Citation](#5-citation)

## 1. Model & Weights

| Model name               | HF Weight |
| ------------------------ | --------- |
| Emu3.5               | [🤗 HF link](https://huggingface.co/BAAI/Emu3.5/tree/main) |
| Emu3.5-Image                | [🤗 HF link](https://huggingface.co/BAAI/Emu3.5-Image/tree/main) |
| Emu3.5-VisionTokenizer     | [🤗 HF link](https://huggingface.co/BAAI/Emu3.5-VisionTokenizer/tree/main) |


*Note:*  
- **Emu3.5** supports general-purpose multimodal predictions, including interleaved image-text generation and single-image generation (T2I/X2I) tasks.
- **Emu3.5-Image** is a model focused on T2I/X2I tasks for best performance on these scenarios.
- Both models are pure next-token predictors without DiDA acceleration (each image may take several minutes to generate).  
- ⚡ **Stay tuned for DiDA-accelerated weights.**

> 💡 **Usage tip:**  
> For **interleaved image-text generation**, use **Emu3.5**.  
> For **single-image generation** (T2I and X2I), use **Emu3.5-Image** for the best quality.



## 2. Quick Start

### Environment Setup

```bash
# Requires Python 3.12 or higher.
git clone https://github.com/baaivision/Emu3.5
cd Emu3.5
pip install -r requirements/transformers.txt
pip install flash_attn==2.8.3 --no-build-isolation
```
### Configuration

Edit `configs/config.py` to set:

- Paths: `model_path`, `vq_path`
  You can use either a **local path** (e.g., downloaded HuggingFace weights) or a **remote HuggingFace Hub ID** for automatic download:
  ```python
  vq_path = "BAAI/Emu3.5-VisionTokenizer"  # remote, auto-download
  model_path = "BAAI/Emu3.5"               # remote, auto-download
  # or
  vq_path = "/path/to/local/Emu3.5-VisionTokenizer"  # local path
  model_path = "/path/to/local/Emu3.5"               # local path
  ```
- Task template: `task_type in {t2i, x2i, howto, story, explore, vla}`
- Input image: `use_image` (True to provide reference images, controls <|IMAGE|> token); set `reference_image` in each prompt to specify the image path. For x2i task, we recommand using `reference_image` as a list containing single/multiple image paths to be compatible with multi-image input.
- Sampling: `sampling_params` (classifier_free_guidance, temperature, top_k/top_p, etc.)
- Aspect Ratio (for t2i task): `aspect_ratio` ("4:3", "21:9", "1:1", "auto" etc..)

### Run Inference

```bash
python inference.py --cfg configs/config.py
```


#### Example Configurations by Task
Below are example commands for different tasks.
Make sure to set CUDA_VISIBLE_DEVICES according to your available GPUs.


```bash
# 🖼️ Text-to-Image (T2I) task
CUDA_VISIBLE_DEVICES=0 python inference.py --cfg configs/example_config_t2i.py

# 🔄 Any-to-Image (X2I) task
CUDA_VISIBLE_DEVICES=0,1 python inference.py --cfg configs/example_config_x2i.py

# 🎯 Visual Guidance task
CUDA_VISIBLE_DEVICES=0,1 python inference.py --cfg configs/example_config_visual_guidance.py

# 📖 Visual Narrative task
CUDA_VISIBLE_DEVICES=0,1 python inference.py --cfg configs/example_config_visual_narrative.py


# After running inference, the model will generate results in protobuf format (.pb files) for each input prompt.
```


Protobuf outputs are written to `outputs/<exp_name>/proto/`. For better throughput, we recommend ≥2 GPUs.


### Run Inference with vLLM

#### vLLM Enviroment Setup

1. [Optional Recommendation] Use a new virtual environment for vLLM backend.
```bash
conda create -n Emu3p5 python=3.12
```

2. Install vLLM and apply the patch files.
```bash
# Requires Python 3.12 or higher.
# Recommended: CUDA 12.8.
pip install -r requirements/vllm.txt
pip install flash_attn==2.8.3 --no-build-isolation

cd Emu3.5
python src/patch/apply.py
```

#### Example Configurations by Task

```bash
# 🖼️ Text-to-Image (T2I) task
CUDA_VISIBLE_DEVICES=0,1 python inference_vllm.py --cfg configs/example_config_t2i.py

# 🔄 Any-to-Image (X2I) task
CUDA_VISIBLE_DEVICES=0,1 python inference_vllm.py --cfg configs/example_config_x2i.py

# 🎯 Visual Guidance task
CUDA_VISIBLE_DEVICES=0,1 python inference_vllm.py --cfg configs/example_config_visual_guidance.py

# 📖 Visual Narrative task
CUDA_VISIBLE_DEVICES=0,1 python inference_vllm.py --cfg configs/example_config_visual_narrative.py
```


### Visualize Protobuf Outputs

To visualize generated protobuf files (--video: Generate video visualizations for interleaved output):

```bash
python src/utils/vis_proto.py --input <input_proto_path> [--output <output_dir>] [--video]
```

- `--input`: supports a single `.pb` file or a directory; directories are scanned recursively.
- `--output`: optional; defaults to `<input_dir>/results/<file_stem>` for files, or `<parent_dir_of_input>/results` for directories.

Expected output directory layout (example):

```text
results/<pb_name>/
├── 000_question.txt
├── 000_global_cot.txt
├── 001_text.txt
├── 001_00_image.png
├── 001_00_image_cot.txt
├── 002_text.txt
├── 002_00_image.png
├── ...
└── video.mp4              # only when --video is enabled
```

Each `*_text.txt` stores decoded segments, `*_image.png` stores generated frames, and matching `*_image_cot.txt` keeps image-level chain-of-thought notes when available.

## 3. Gradio Demo

We provide two Gradio Demos for different application scenarios:

 Emu3.5-Image Demo —— Interactive interface optimized for Text-to-Image (T2I) and Any-to-Image (X2I) tasks:

```bash
CUDA_VISIBLE_DEVICES=0,1 python gradio_demo_image.py --host 0.0.0.0 --port 7860
```

Emu3.5-Interleave Demo —— Launch Emu3.5 Interleave Tasks (Visual Guidance and Visual Narrate) Gradio Demo
```bash
CUDA_VISIBLE_DEVICES=0,1 python gradio_demo_interleave.py --host 0.0.0.0 --port 7860
```

### Features

- Image Generation: Support Text-to-Image Generation and Multimodal Image Generation
- Interleaved Generation: Support long-sequence creation with alternating image and text generation
- Multiple Aspect Ratios for T2I: 9 preset aspect ratios (4:3, 16:9, 1:1, etc.) plus auto mode
- Chain-of-Thought Display: Automatically parse and format model's internal thinking process
- Real-time Streaming: Stream text and image generation with live updates

### Official Web & Mobile Apps

- **Web**: Production-ready Emu3.5 experience is available at [zh.emu.world](https://zh.emu.world) (Mainland China) and [emu.world](https://emu.world) (global), featuring a curated homepage, “Create” workspace, inspiration feed, history, personal profile, and language switching.
- **Mobile (Android APK & H5)**: Mobile clients provide the same core flows — prompt-based creation, “inspiration” gallery, personal center, and feedback & privacy entrypoints — with automatic UI language selection based on system settings.
- **Docs**: For product usage details, see the **Emu3.5 AI 使用指南 (Chinese)** and **Emu3.5 AI User Guide (English)**:  
  - CN: [Emu3.5 AI 使用指南](https://jwolpxeehx.feishu.cn/wiki/BKuKwkzZOi4pdRkVV13csI0FnIg?from=from_copylink)  
  - EN: [Emu3.5 AI User Guide](https://jwolpxeehx.feishu.cn/wiki/Gcxtw9XHhisUu8kBEaac6s6xnhc?from=from_copylink)

#### Mobile App Download (QR Codes)

<div align='center'>
  <table>
    <tr>
      <td align="center">
        <img src="./assets/qr_zh.png" alt="Emu3.5 Mobile App (Mainland China)" width="220" />
        <br />
        <sub><b>Emu3.5 Mobile · Mainland China</b></sub>
      </td>
      <td align="center">
        <img src="./assets/qr.png" alt="Emu3.5 Mobile App (Global)" width="220" />
        <br />
        <sub><b>Emu3.5 Mobile · Global</b></sub>
      </td>
    </tr>
  </table>
</div>



## 4. Schedule

- [x] Inference Code (NTP Version)
- [ ] Advanced Image Decoder
- [ ] Discrete Diffusion Adaptation (DiDA) Inference & Weights


## 5. Citation

```bibtex
@misc{cui2025emu35nativemultimodalmodels,
      title={Emu3.5: Native Multimodal Models are World Learners}, 
      author={Yufeng Cui and Honghao Chen and Haoge Deng and Xu Huang and Xinghang Li and Jirong Liu and Yang Liu and Zhuoyan Luo and Jinsheng Wang and Wenxuan Wang and Yueze Wang and Chengyuan Wang and Fan Zhang and Yingli Zhao and Ting Pan and Xianduo Li and Zecheng Hao and Wenxuan Ma and Zhuo Chen and Yulong Ao and Tiejun Huang and Zhongyuan Wang and Xinlong Wang},
      year={2025},
      eprint={2510.26583},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.26583}, 
}
```


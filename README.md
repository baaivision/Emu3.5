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


*Note:*
- **Emu3.5** supports general tasks, including interleaved generation and image generation/editing.  
- **Emu3.5-Image** focuses on high-quality image generation and editing.  
- Both **Emu3.5** and **Emu3.5-Image** are currently pure next-token prediction models without DiDA acceleration. Stay tuned for DiDA weights.


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
- Task template: `task_type in {t2i, x2i, howto, story, explore, vla}`
- Input image: `use_image` (True to provide reference images, controls <|IMAGE|> token); set `reference_image` in each prompt to specify the image path. For x2i task, 'reference_image' should be a list rather than a single path to be compatible for multi-image input.
- Sampling: `sampling_params` (classifier_free_guidance, temperature, top_k/top_p, etc.)

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

# 🌍 World Exploration task
CUDA_VISIBLE_DEVICES=0,1 python inference.py --cfg configs/example_config_world_exploration.py

# After running inference, the model will generate results in protobuf format (.pb files) for each input prompt.
```


Protobuf outputs are written to `outputs/<exp_name>/proto/`. For better throughput, we recommend ≥2 GPUs.

### Visualize Protobuf Outputs

To visualize generated protobuf files:

```bash
python src/utils/vis_proto.py --input <input_proto_file> --output <output_dir>
```

## 3. Schedule

- [x] Inference Code (NTP Version)
- [ ] Advanced Image Decoder
- [ ] Discrete Diffusion Adaptation (DiDA) Inference & Weights


## 4. Citation

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


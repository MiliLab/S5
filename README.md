# 🌍 S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing

This repository provides the official implementation, datasets, and checkpoints for **S5**, the first scalable semi-supervised semantic segmentation framework in remote sensing.

------

## 🎯 Introduction

- **Dataset (RS4P-1M)**:
   We curate RS4P-1M, a large-scale dataset with 1 million unlabeled remote sensing images with pseudo-labels.
- **S4P (Semi-supervised Semantic Segmentation Pre-training)**:
   Extends traditional semi-supervised semantic segmentation (S4) into large-scale pre-training, leveraging RS4P-1M with FixMatch to learn generalizable representations.
- **MoE-MDF (Mixture-of-Experts Multi-Dataset Fine-tuning)**:
   A multi-dataset fine-tuning strategy with shared + task-specific experts, enabling efficient adaptation across RS benchmarks with minimal overhead.

------

## ✅ To-do List

-  Release checkpoints of S5 (ViT-B/L/H).
-  Release pre-training codes and configs for S4P.
-  Release RS4P-1M dataset.
-  Release codes and configs for downstream tasks (Object Detection, Semantic Segmentation).

------

## 🔥 News

- **2025.08**: Paper released on [arXiv](https://arxiv.org/abs/2508.12409).
- **2025.08**: We released the S4P code and the pretrained weights (ViT-B/L). Download link: [Baidu Netdisk](https://pan.baidu.com/s/1MC3moItUZvriXFeKj7I2jA), extraction code: `huuh`.

## 📚 Contents

- [Performance](#performance)
- [RS4P-1M](#rs4p-1m)
- [S4P](#s4p)
- [MoE-MDF](#moe-mdf)
- [License](#license)

------

## 📊 Performance

We compare S5 with state-of-the-art remote sensing foundation models (RSFMs) on **semantic segmentation** and **object detection** benchmarks.

| Method        | Backbone     | Params Det (M, Single) | Params Det (M, Multiple) | DIOR-R    | DOTA-v2   | Params Seg (M, Single) | Params Seg (M, Multiple) | Vaihingen | Potsdam   | LoveDA    | OpenEarthMap |
| ------------- | ------------ | ---------------------- | ------------------------ | --------- | --------- | ---------------------- | ------------------------ | --------- | --------- | --------- | ------------ |
| RVSA          | ViT-B + RVSA | 111.2                  | 222.4                    | 68.06     | 55.22     | 103.2                  | 412.8                    | 78.49     | 91.58     | 52.44     | 66.63        |
| GFM           | Swin-B       | 104.1                  | 208.2                    | 67.67     | 59.15     | 96.9                   | 387.6                    | 79.61     | 91.85     | 54.98     | 67.78        |
| Scale-MAE     | ViT-L        | 334.6                  | 669.2                    | 66.47     | 56.97     | 327.4                  | 1309.6                   | 78.64     | 91.54     | 53.67     | 68.54        |
| SAMRS         | ViT-B + RVSA | -                      | -                        | -         | -         | 103.2                  | 412.8                    | 78.73     | 91.69     | 53.04     | 67.37        |
| SatMAE++      | ViT-L        | 334.6                  | 669.2                    | 66.82     | 55.60     | 327.4                  | 1309.6                   | 78.80     | 91.64     | 52.82     | 65.62        |
| BillionFM     | ViT-G        | 996.9                  | 1993.9                   | 73.62     | 58.69     | 990.9                  | -                        | -         | 92.58     | 54.40     | -            |
| OREOLE        | ViT-G        | 996.9                  | -                        | 71.31     | -         | 990.9                  | -                        | -         | 92.20     | 54.00     | -            |
| MTP           | ViT-L + RVSA | 334.6                  | 669.2                    | 74.54     | 58.41     | 327.4                  | 1309.6                   | 80.62     | 92.47     | 54.16     | 69.04        |
| MA3E          | ViT-B        | 111.2                  | -                        | 71.82     | -         | 103.2                  | -                        | -         | 91.50     | -         | -            |
| SelectiveMAE  | ViT-L        | 334.6                  | 669.2                    | 71.75     | 57.84     | 327.4                  | 1309.6                   | 80.45     | 92.78     | 54.31     | 69.30        |
| **S5 (Ours)** | ViT-B        | 111.2                  | 138.3                    | 72.95     | 57.20     | 103.2                  | 160.4                    | 79.85     | 92.40     | 54.02     | 68.65        |
| **S5 (Ours)** | ViT-L        | 334.6                  | 377.8                    | 75.21     | 59.71     | 327.4                  | 435.0                    | 80.72     | 92.78     | **55.67** | 69.66        |
| **S5 (Ours)** | ViT-H        | 671.7                  | 730.0                    | **75.30** | **59.89** | 663.4                  | 824.5                    | **80.85** | **92.97** | 55.65     | **70.02**    |

------

## 🚀 RS4P-1M

RS4P-1M is a large-scale optical remote sensing dataset for semi-supervised semantic segmentation pre-training, comprising one million images with high-quality pseudo-labels.

------

## 🚀 S4P

------

## 🚀 MoE-MDF

Unified fine-tuning across multiple RS benchmarks with shared + task-specific experts.
 Supports semantic segmentation (Vaihingen, Potsdam, LoveDA, OpenEarthMap) and object detection (DIOR-R, DOTA-v2.0).

------

# ⭐ Citation

If you find S5 helpful, please consider giving this repo a ⭐ and citing:

```latex
@article{S5,
  title={S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing},
  author={Liang Lv and Di Wang and Jing Zhang and Lefei Zhang},
  journal={arXiv preprint arXiv:2508.12409},
  year={2025}
}
```

## 🤝 License

Apache License 2.0. Please check [LICENSE.md](https://chatgpt.com/c/docs/LICENSE.md) for details.

------

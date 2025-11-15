  <h2 align="center"><strong>S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing</strong></h2>

<div align="center">
<h5>
<em>Liang Lv<sup>1</sup>, Di Wang<sup>1,2</sup>, Jing Zhang<sup>1 †</sup>, Lefei Zhang<sup>1 †</sup></em>
    <br><br>
        <sup>1</sup> National Engineering Research Center for Multimedia Software, School of Computer Science, Wuhan University, China<br/> 
        <sup>2</sup> Zhongguancun Academy, China
</h5>
</div>


<h5 align="center">
<a href="https://arxiv.org/pdf/2508.12409"> <img src="https://img.shields.io/badge/Arxiv-2508.12409-b31b1b.svg?logo=arXiv"></a>
</h5>

------

## 🎯 Introduction

**S5** is a scalable semi-supervised learning framework designed for remote sensing semantic segmentation and oriented object detection. It includes three core contributions:

- **Dataset (RS4P-1M):**

  We curate RS4P-1M, a large-scale dataset with 1 million unlabeled remote sensing images with pseudo-labels.

- **S4P (Semi-supervised Semantic Segmentation Pre-training):**

  Extends traditional semi-supervised semantic segmentation (S4) into large-scale pre-training, leveraging RS4P-1M with FixMatch to learn generalizable representations.

- **MoE-MDF (Mixture-of-Experts Multi-Dataset Fine-tuning):**

  A multi-dataset fine-tuning strategy with shared + task-specific experts, enabling efficient adaptation across RS benchmarks with minimal overhead.

------

## 🔥 News

- **2025.08**: Paper released on [arXiv](https://arxiv.org/abs/2508.12409).

- **2025.08**\: We released the S4P code and the pretrained weights (ViT-B/L). Download link: [Baidu Netdisk](https://pan.baidu.com/s/1MC3moItUZvriXFeKj7I2jA), extraction code: `huuh`.

- **2025.09**: We released the fine-tuning code and weights for remote sensing semantic segmentation (ViT-B/L). Download link: [Baidu Netdisk](https://pan.baidu.com/s/1omtC2Lgrv9MZfMWmourA9g), extraction code: `4xvx`.

- **2025.09**: We released the fine-tuning code and weights for remote sensing rotated object detection (ViT-B/L). Download link: [Baidu Netdisk](https://pan.baidu.com/s/13j5WuYEC6FfhuRpko_9epw), extraction code: `y9s3`.

- **2025.11**: S5 has been accepted as an **Oral** paper at **AAAI 2026**!

------

## 📚 Contents

- [Performance](https://chatgpt.com/c/691834fd-a5a8-832e-9c58-83ca5cb2f654#performance)
- [RS4P-1M](https://chatgpt.com/c/691834fd-a5a8-832e-9c58-83ca5cb2f654#rs4p-1m)
- [S4P](https://chatgpt.com/c/691834fd-a5a8-832e-9c58-83ca5cb2f654#s4p)
- [MoE-MDF](https://chatgpt.com/c/691834fd-a5a8-832e-9c58-83ca5cb2f654#moe-mdf)
- [License](https://chatgpt.com/c/691834fd-a5a8-832e-9c58-83ca5cb2f654#license)

------

## 📊 Performance

We compare S5 against state-of-the-art Remote Sensing Foundation Models (RSFMs) on both **semantic segmentation** and **oriented object detection** tasks.

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

## 🚀 S4P (Semi-supervised Semantic Segmentation Pre-training)

S4P extends semi-supervised segmentation into the large-scale setting, leveraging FixMatch and ViT backbones to learn strong visual representations on RS4P-1M.

### :gear: Installation for Pretraining

```sh
conda create -n s5_seg python=3.10 -y
conda activate s5_seg

# Install PyTorch
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install additional dependencies
pip install -r requirements.txt
```

### 🚙 Start Pretraining (Example: ViT-B)

1. Download the RS4P-1M dataset
2. Set `data_root` in `S4_Pretrain/configs/pretrain.yaml`
3. Run in the `S5/S4_Pretrain/scripts` directory:

```sh
bash train.sh 8 12345 vit_b mae
```

------

## 🚀 MoE-MDF: Multi-Dataset Fine-tuning with Mixture-of-Experts

Unified fine-tuning across multiple RS benchmarks with shared + task-specific experts.

Supports semantic segmentation (Vaihingen, Potsdam, LoveDA, OpenEarthMap) and object detection (DIOR-R, DOTA-v2.0).

### :gear: Installation for Fine-tuning

Semantic segmentation uses the same environment as S4P.

For the object detection task, we build upon and modify the OBBDetection. The detailed initialization and configuration procedures can be found in the official documentation:[OBBDetection Installation Guide](https://github.com/jbwang1997/OBBDetection/blob/master/docs/install.md). The main runtime environment and dependencies of our project are as follows:

```sh
conda create -n s5_det python=3.8.20 -y
conda activate s5_det

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install mmcv-full==1.3.16
pip install mmengine==0.10.7
```

------

## 🧩 Semantic Segmentation Fine-tuning

After preparing all datasets (Vaihingen, Potsdam, LoveDA, OpenEarthMap), run the following commands in the `S5/Semantic_Segmentation/scripts/` directory:

```sh
bash md_finetune.sh 2 1156 vit_b_moe True Your/Path/vit_b_s4p_upernet
```

------

## 🛩 Oriented Object Detection Fine-tuning

Prepare DIOR-R and DOTA-v2.0, then run in `S5/Object_detection`:

```sh
CUDA_VISIBLE_DEVICES="0,1,2,3" \
python -m torch.distributed.launch \
  --nproc_per_node=6 --master_port=12345 \
  tools/train.py ./configs/obb/oriented_rcnn/mtd/vit_b_moe_dior_r_dota2.py \
  --launcher 'pytorch' \
  --options 'find_unused_parameters'=False
```

------

## ⭐ Citation

If you find S5 helpful, please consider ⭐ starring the repo and citing our paper:

```latex
@article{S5,
  title={S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing},
  author={Liang Lv and Di Wang and Jing Zhang and Lefei Zhang},
  journal={arXiv preprint arXiv:2508.12409},
  year={2025}
}
```

------

## 🤝 License

Apache License 2.0. Please check [LICENSE.md](https://chatgpt.com/c/docs/LICENSE.md) for details.
<div align="center">

<h1>Disentangle-Then-Distill: Complementary Expert-Guided Structured Distillation<br>for Remote Sensing Semantic Segmentation</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.7.1"/>
  <img src="https://img.shields.io/badge/Task-Semantic%20Segmentation-4C8BF5" alt="Semantic Segmentation"/>
  <a href="https://pan.baidu.com/s/1XAEgz_Gyi-DZ_Ec_gN3cUg?pwd=8tbc"><img src="https://img.shields.io/badge/Weights-Available-2EA44F" alt="Weights"/></a>
</p>

<p>
  <a href="#highlights">Highlights</a> |
  <a href="#architecture">Architecture</a> |
  <a href="#installation">Installation</a> |
  <a href="#training">Training</a> |
  <a href="#results">Results</a>
</p>

</div>

---

## Highlights

- **Disentangle then distill.** GeoMoE uses TPE and AGE to organize complementary texture-oriented and geometry-oriented knowledge before distillation.
- **Structured knowledge transfer.** SCD, CPD, and RCRD transfer shared semantics, expert-specific knowledge, and reliable class relations.
- **Efficient optical-only inference.** CESDNet-KD requires no nDSM at inference and achieves 82.32% / 77.03% mIoU on Vaihingen / Potsdam with 5.39 M trainable parameters.

## Architecture

### Overall Framework

<p align="center">
  <img src="./images/CESDNet.jpg" alt="Overall architecture of CESDNet" width="96%"/>
</p>

<p align="center"><sub><strong>Figure 1.</strong> Overall framework of CESDNet.</sub></p>

### Multimodal Teacher: CESDNet-T

<p align="center">
  <img src="./images/CESDNet-T.jpg" alt="Architecture of CESDNet-T" width="96%"/>
</p>

<p align="center"><sub><strong>Figure 2.</strong> Architecture of the dual-expert multimodal teacher CESDNet-T.</sub></p>

## Environment

### Requirements

- Python `3.11`
- PyTorch `2.7.1`
- torchvision `0.22.1`
- mmcv-lite `2.1.0`
- CUDA-enabled GPU for training

### Installation

```bash
conda create -n cesdnet python=3.11 -y
conda activate cesdnet
pip install -r requirements.txt
```

### Teacher Backbone

- Standard DINOv3 ViT-L/16 pretrained on LVD-1689M web images
- Reported trainable parameters exclude the frozen DINOv3 backbone

## Pretrained Models

- **CESDNet weights:** [Baidu Netdisk](https://pan.baidu.com/s/1XAEgz_Gyi-DZ_Ec_gN3cUg?pwd=8tbc) (extraction code: `8tbc`)
- **Teacher backbone:** local DINOv3 repository and ViT-L/16 checkpoint
- **Student backbone:** optional DFormer-Tiny checkpoint configured by `CESDNetConfig.dformer_weight`

```text
checkpoints/
|-- CESDNet-T-Potsdam.pth
|-- CESDNet-KD-Potsdam.pth
`-- dinov3_vitl16_pretrain.pth
```

The DINOv3 repository and checkpoint are required when constructing CESDNet-T. If `CESDNetConfig.dformer_weight` is empty, no DFormer pretrained weights are loaded.

## Data Preparation

Run all commands from the directory containing `CESDNet/`.

- `database/<dataset>/`: patch data used for training; metadata is read from `configs/Potsdam.json` or `configs/Vaihingen.json`.
- `ISPRS/`: original full-resolution images and labels used by full-tile testing and evaluation.

```text
project-root/
|-- CESDNet/
|-- configs/
|-- database/
|-- ISPRS/
|-- toolbox/
`-- log.py
```

## Training

Paper settings:

- Optimizer: AdamW
- Epochs: `100`
- Learning rate: `1e-4`
- Weight decay: `5e-4`
- Random seed: `3407`
- Batch size: `8` for CESDNet-T; `16` for CESDNet-S and CESDNet-KD

```bash
# Train CESDNet-T
python CESDNet/train_teacher.py \
  --dataset Potsdam \
  --teacher-repo path/to/dinov3 \
  --teacher-backbone checkpoints/dinov3_vitl16_pretrain.pth \
  --amp

# Train CESDNet-S
python CESDNet/train_student.py --dataset Potsdam --amp

# Train CESDNet-KD
python CESDNet/train_kd.py \
  --dataset Potsdam \
  --teacher-checkpoint checkpoints/CESDNet-T-Potsdam.pth \
  --teacher-repo path/to/dinov3 \
  --teacher-backbone checkpoints/dinov3_vitl16_pretrain.pth \
  --amp
```

Replace `Potsdam` with `Vaihingen` for the other benchmark.

## Testing and Evaluation

```bash
python CESDNet/test.py \
  --dataset Potsdam \
  --model kd \
  --weights checkpoints/CESDNet-KD-Potsdam.pth \
  --mode tile \
  --stride 32 \
  --output-dir results/Potsdam/CESDNet-KD

python CESDNet/eval.py \
  --dataset Potsdam \
  --pred-dir results/Potsdam/CESDNet-KD \
  --metric-mode paper
```

In `paper` mode, metrics are computed per tile and then averaged. Clutter/background is excluded from the headline Vaihingen metrics, while all six classes are evaluated on Potsdam.

## Results

Results use `256 x 256` inputs. **Trainable Params** excludes the frozen DINOv3 backbone.

<table>
  <thead>
    <tr>
      <th align="center" rowspan="2"><small>Model</small></th>
      <th align="center" rowspan="2"><small>Input</small></th>
      <th align="center" rowspan="2"><small>Trainable<br>Params (M)</small></th>
      <th align="center" rowspan="2"><small>FLOPs (G)</small></th>
      <th align="center" colspan="2"><small>Vaihingen</small></th>
      <th align="center" colspan="2"><small>Potsdam</small></th>
    </tr>
    <tr>
      <th align="center"><small>mAcc (%)</small></th>
      <th align="center"><small>mIoU (%)</small></th>
      <th align="center"><small>mAcc (%)</small></th>
      <th align="center"><small>mIoU (%)</small></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><small>CESDNet&#8209;T</small></td>
      <td align="center"><small>Optical + nDSM</small></td>
      <td align="center"><small>100.04</small></td>
      <td align="center"><small>76.54</small></td>
      <td align="center"><small><strong>90.63</strong></small></td>
      <td align="center"><small><strong>83.71</strong></small></td>
      <td align="center"><small><strong>87.22</strong></small></td>
      <td align="center"><small><strong>78.95</strong></small></td>
    </tr>
    <tr>
      <td align="center"><small>CESDNet&#8209;S</small></td>
      <td align="center"><small>Optical only</small></td>
      <td align="center"><small><strong>5.39</strong></small></td>
      <td align="center"><small><strong>2.10</strong></small></td>
      <td align="center"><small>86.10</small></td>
      <td align="center"><small>77.47</small></td>
      <td align="center"><small>83.47</small></td>
      <td align="center"><small>73.97</small></td>
    </tr>
    <tr>
      <td align="center"><small><strong>CESDNet&#8209;KD</strong></small></td>
      <td align="center"><small><strong>Optical only</strong></small></td>
      <td align="center"><small><strong>5.39</strong></small></td>
      <td align="center"><small><strong>2.10</strong></small></td>
      <td align="center"><small>89.86</small></td>
      <td align="center"><small>82.32</small></td>
      <td align="center"><small>85.54</small></td>
      <td align="center"><small>77.03</small></td>
    </tr>
  </tbody>
</table>

## Citation

The paper link and BibTeX will be updated upon publication.

## Acknowledgement

This project builds on [DINOv3](https://github.com/facebookresearch/dinov3) and [DFormer](https://github.com/VCIP-RGBD/DFormer).

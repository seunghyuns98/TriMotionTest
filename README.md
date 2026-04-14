<div align="center">

# TriMotion: Modality-Agnostic Camera Control for Video Generation

<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-red)](#) -->
<!-- [![Project Page](https://img.shields.io/badge/Project-Page-blue)](#) -->
<!-- [![Demo](https://img.shields.io/badge/Demo-HuggingFace-yellow)](#) -->
<!-- [![Dataset](https://img.shields.io/badge/Dataset-GoogleDrive-green)](#) -->

<!-- **ECCV 2026** -->

*A unified framework for camera-controlled video generation that accepts **video**, **pose**, or **text** — all describing the same camera trajectory — and maps them into a shared motion embedding space.*

</div>

<!-- <p align="center">
  <img src="assets/teaser.gif" width="90%">
</p> -->

---

## 📢 News

<!-- - **[2026-03]** TriMotion is accepted to ECCV 2026. 🎉 -->
- **[2026-04]** Code, checkpoints, and the **Motion Triplet Dataset** are released.

---

## 🔍 Overview

Existing camera-control methods are typically restricted to a single input modality — pose-conditioned methods require precise geometric trajectories, reference-video methods lack explicit control, and text-based methods struggle with temporal consistency. **TriMotion** addresses all three limitations in one framework.

**Key contributions**

1. **Unified Motion Embedding Space** — aligns video, pose, and text in a shared representation via contrastive learning, temporal synchronization, and geometric fidelity regularization.
2. **Motion Triplet Dataset** — 136K synchronized *(video, pose, text)* triplets built on top of the MultiCamVideo Dataset with LLM-generated, geometry-grounded captions.
3. **Latent Motion Consistency** — a Motion Embedding Predictor that enforces trajectory fidelity directly in latent space, avoiding costly pixel-space decoding.

Built on top of [Wan2.1](https://github.com/Wan-Video/Wan2.1) and supports both **I2V** and **V2V** camera-controlled generation.

---

## ⚙️ Installation

Tested with **Python 3.10**, **PyTorch 2.x**, **CUDA 11.8+**.

```bash
git clone https://github.com/<your-org>/TriMotion.git
cd TriMotion

pip install -r requirements.txt
```

---

## 📦 Download Pretrained Weights

We use [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) as the diffusion backbone, along with an additional CLIP checkpoint required for the I2V branch.

```bash
# 1) Wan2.1-T2V-1.3B (T5 text encoder, VAE, DiT)
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir checkpoint/Wan2.1-T2V-1.3B

# 2) CLIP image encoder (open-clip-xlm-roberta-large-vit-huge-14)
hf download DeepBeepMeep/Wan2.1 \
    models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --revision 8bee6e003d1d9d31ecb2c75b643e57fa74fb2ad5 \
    --local-dir ./checkpoint/Wan2.1-T2V-1.3B
```

After downloading, the `checkpoint/` directory should look like:

```
checkpoint/
└── Wan2.1-T2V-1.3B/
    ├── models_t5_umt5-xxl-enc-bf16.pth
    ├── Wan2.1_VAE.pth
    ├── diffusion_pytorch_model.safetensors
    └── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
```

You also need the TriMotion-specific checkpoints (Stage 1 / Stage 2 / Stage 3), available from our [Google Drive folder](https://drive.google.com/drive/folders/1tQznlZwoSTFRzDhgmikVCGAbDiO6YAVs?usp=sharing). Download the entire folder with [`gdown`](https://github.com/wkentaro/gdown):

```bash
pip install gdown

# Download the whole TriMotion checkpoint folder into ./checkpoint/trimotion/
gdown --folder https://drive.google.com/drive/folders/1tQznlZwoSTFRzDhgmikVCGAbDiO6YAVs \
      -O ./checkpoint/trimotion
```

> 💡 If `gdown` hits a quota error for large files, re-run the same command — partially downloaded files will be resumed. For very large files you may need `gdown --fuzzy <file-url>` on individual items.

| Checkpoint | Description |
|---|---|
| Stage 1 — Unified Motion Embedding Space | Video / text / pose encoders |
| Stage 2 — Motion Embedding Predictor | Latent → motion embedding |
| Stage 3 — Wan2.1 Fine-tuned DiT | Camera-controlled I2V / V2V |
| VGGT — [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | Video motion encoder backbone |

---

## 🚀 Inference

### Multimodal demo (single example)

```bash
python demo_multimodal_single.py \
    --content_video  path/to/source.mp4 \
    --ref_video      path/to/ref.mp4 \
    --prompt         "A cinematic shot of a snowy mountain at sunset." \
    --camera_caption "The camera tilts up very fast." \
    --output_dir     ./results/single \
    --mode           v2v
```

Key options:

- `--scale` — blend weight between video-based and text-based camera trajectory (0 = text only, 1 = ref video only).
- `--mode` — `i2v` or `v2v`.
- `--first_frame` — optional image path for I2V mode.

### Batch / CSV demo

```bash
python demo_mutlimodal.py \
    --dataset_path demo/example_csv/infer/example_camclone_testset.csv \
    --output_dir   ./results/batch \
    --mode         v2v
```

---

---

## 🎬 Motion Triplet Dataset

We release the **Motion Triplet Dataset**, built upon the [MultiCamVideo Dataset](https://github.com/KlingAIResearch/ReCamMaster) (136K videos, 13.6K scenes, 40 Unreal Engine 5 environments) by adding geometry-grounded motion descriptions.

**Preparation**

1. Download the **MultiCamVideo Dataset** into the `MotionTriplet-Dataset/` directory.
2. Download our **Motion Descriptions** from [Google Drive](https://drive.google.com/file/d/1k4c7M6ttohMEXq1EVn7PARg0f8M7fLBA/view?usp=sharing).
3. Run the preparation script:

```bash
python preparing_dataset.py
```

**Directory structure**

```
MotionTriplet-Dataset/
├── train/
│   └── f00/
│       └── scene1/
│           ├── cameras/
│           │   ├── camera_extrinsics.json
│           │   ├── text_description_long.json
│           │   └── text_description_short.json
│           ├── videos/
│           │   ├── cam01.mp4
│           │   ├── ...
│           │   └── cam10.mp4
│           ├── text/
│               └── text_description.json   
│           └── merged_conditions.json
```

### Preprocess Embeddings

Precompute and cache embeddings before training:

```bash
python latent_preprocess.py \
    --base_path ./data \
    --json_path ./data/merged_camera_dataset.json \
    --t5_path path/to/t5-base \
    --vggt_path path/to/vggt \
    --embedding_model_path path/to/stage1_checkpoint \
    --output_dir ./data
```

---

## Training

### Stage 1: Unified Motion Embedding Space

Trains motion encoders for all three modalities with a composite loss: global InfoNCE alignment, temporal synchronization, and geometric fidelity regularization.

```bash
python train_embedding_space.py \
    --base_path ./data \
    --json_path ./data/merged_camera_dataset.json \
    --t5_path path/to/t5-base \
    --vggt_path path/to/vggt \
    --output_dir ./checkpoint/stage1 \
    --batch_size 16 \
    --lr 1e-4 \
    --epochs 100
```

### Stage 2: Motion Embedding Predictor

Trains the predictor (3D convolutions + temporal Transformer) to estimate motion embeddings from VAE latents, using a dual-granularity cosine similarity loss (global + frame-wise).

```bash
python train_motion_embedding_projector.py \
    --base_path ./data \
    --json_path ./data/merged_camera_dataset.json \
    --embedding_model_path ./checkpoint/stage1/best.ckpt \
    --output_dir ./checkpoint/stage2 \
    --batch_size 24 \
    --lr 1e-4 \
    --epochs 10
```

### Stage 3: Diffusion Model Fine-tuning

Fine-tunes WAN-Video with motion embedding conditioning via block-specific projection MLPs. Jointly trains I2V and V2V with equal probability per iteration.

```bash
deepspeed train_TriMotion.py \
    --base_path ./data \
    --json_path ./data/merged_camera_dataset.json \
    --wan_model_path path/to/wan-video \
    --t5_path path/to/t5-base \
    --embedding_model_path ./checkpoint/stage1/best.ckpt \
    --projector_path ./checkpoint/stage2/best.ckpt \
    --output_dir ./checkpoint/stage3 \
    --batch_size 4 \
    --lr 1e-4 \
    --deepspeed_stage 2
```

Training was performed on **4 × NVIDIA H200 GPUs** with AdamW (β₁=0.9, β₂=0.999, weight decay=0.01, lr=1×10⁻⁴).

---

## Inference

```bash
python demo_multimodal.py \
    --wan_model_path path/to/wan-video \
    --embedding_model_path ./checkpoint/stage1/best.ckpt \
    --projector_path ./checkpoint/stage2/best.ckpt \
    --stage3_path ./checkpoint/stage3 \
    --input_video path/to/reference.mp4 \
    --prompt "The camera starts with a steady dolly-in motion while gradually panning left." \
    --output_path output.mp4
```

---

## Method

### Unified Motion Embedding Space

Each modality encoder produces `N` temporal motion tokens + 1 global token, processed by a lightweight temporal Transformer `T_m`:

| Modality | Encoder | Key design |
|---|---|---|
| Video | VGGT Aggregator (Alternating-Attention blocks) | Camera tokens aggregate multi-view 3D geometry |
| Text | Frozen T5 + N learnable motion queries (cross-attention) | Lifts static text into temporal motion sequence |
| Pose | Frame-wise MLP (GELU) on flattened 3×4 extrinsic matrix | Preserves geometric trajectory structure |

**Training objectives:**

- **L_NCE** (Global Alignment): InfoNCE contrastive loss over all 3 modality pairs
- **L_temp** (Temporal Synchronization): Cosine distance between corresponding temporal tokens
- **L_pose** (Geometric Fidelity): Shared pose regressor predicting camera extrinsics from each modality embedding (L1 loss)

```
L_align = L_NCE + λ_t · L_temp + λ_p · L_pose
```

### Latent Motion Consistency

A frozen Motion Embedding Predictor `M_pred` estimates a motion embedding from the reconstructed clean latent during diffusion training:

```
ẑ_0 = z_t - t · v_θ(z̃_t, t, y, I, e_m)
L_total = L_denoise + λ_m · L_motion(M_pred(ẑ_0), e_m)
```

This enforces trajectory adherence without pixel-space decoding.

### Motion Conditioning

Motion embeddings are injected into each DiT block via a block-specific projection MLP with residual addition:

```
h_in = h + F_proj(e_m)
```

Only the 3D spatial-temporal attention layers and projection MLPs are updated during fine-tuning.

---

## Applications

### Sequential Motion Composition

Concatenate two motion sequences (offset by the final state of the first) to generate compound multi-stage camera trajectories across any modality combination.

### Cross-Modal Motion Interpolation

Linearly interpolate between motion embeddings `e_a` and `e_b` from different modalities to produce smooth blended camera motion.

---

## Citation

```bibtex
@inproceedings{trimotion2026,
  title={TriMotion: Modality-Agnostic Camera Control for Video Generation},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

---

## Acknowledgements

- [WAN-Video](https://github.com/Wan-Video/Wan2.1) — diffusion backbone
- [VGGT](https://github.com/facebookresearch/vggt) — video motion encoder
- [ReCamMaster](https://github.com/jianhongbai/ReCamMaster) — Multi-Cam Video Dataset
- [Qwen3](https://github.com/QwenLM/Qwen3) — geometry-grounded caption generation
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — T5 text encoder

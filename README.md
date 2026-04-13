# TriMotion: Modality-Agnostic Camera Control for Video Generation

<!-- > **ECCV 2026** | [Paper](#) | [Project Page](#) | [Demo](#) -->

TriMotion is a unified framework for camera-controlled video generation that maps **video, pose, and text** inputs describing the same camera trajectory into a **shared motion embedding space**. This modality-agnostic design enables flexible and consistent camera control from heterogeneous user inputs, built on top of [WAN-Video](https://github.com/Wan-Video/Wan2.1).

---

## Overview

Existing camera-control methods are typically restricted to a single input modality — pose-conditioned methods require precise geometric trajectories, reference-video methods lack explicit control, and text-based methods struggle with temporal consistency. TriMotion addresses all three limitations in a single framework.

**Key components:**
1. **Unified Motion Embedding Space** — aligns video, pose, and text in a shared representation via contrastive learning, temporal synchronization, and geometric fidelity regularization
2. **Motion Triplet Dataset** — 136K synchronized (video, pose, text) triplets built on the Multi-Cam Video Dataset with LLM-generated geometry-grounded captions
3. **Latent Motion Consistency** — a Motion Embedding Predictor that enforces trajectory fidelity directly in latent space, avoiding costly pixel-space decoding

**Three-Stage Training Pipeline:**

```
Stage 1: Train Unified Motion Embedding Space  (video + text + pose alignment)
    ↓
Stage 2: Train Motion Embedding Predictor      (latent → motion embedding)
    ↓
Stage 3: Train WAN-Video Diffusion Model   (camera-controlled I2V / V2V)
```

---

## Requirements

```bash
pip install torch torchvision
pip install transformers diffusers accelerate deepspeed
pip install pytorch-lightning
pip install decord einops scipy pillow numpy
```

Tested with Python 3.10, PyTorch 2.x, CUDA 11.8+.

---

## Motion Triplet Dataset

The Motion Triplet Dataset is built upon the Multi-Cam Video Dataset(136K videos, 13.6K scenes, 40 Unreal Engine 5 environments) by adding geometry-grounded motion descriptions.

1. First, download Multi-Cam Video Dataset [Multi-Cam Video Dataset](https://github.com/KlingAIResearch/ReCamMaster?tab=readme-ov-file) under MotionTriplet-Dataset directory.
2. Then download Motion Descriptions from [Google Drive](https://drive.google.com/file/d/1VD-9rAHo1vJH_Vtx0NOFGzeDr5TfBh6I/view?usp=sharing) and put it under MotionTriplet-Dataset directory.
3. Finally, run below code to prepare dataset for training.
```bash
python merge_datasets.py
```
The structure of Full Dataset would be as below:

### Directory Structure

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

### Metadata Format (`merged_conditions.json`)

```json
{
    "Cam01": {
        "extrinsics": {
            [[4x4 matrix per frame], ...]
        }
        "captions": {
            "long": "The camera begins with a steady counterclockwise rotation, forming an orbiting motion...",
            "short": "Orbit left, then dolly forward",
            "text": "The video showcases a dimly lit, nocturnal suburban scene."
        }
    }
    ...
}
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

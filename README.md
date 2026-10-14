# HMCG-UIC

Reproducible implementation of **Human-Machine Collaborative Generative Underwater Image Compression**. HMCG-UIC is built on a retained and adapted StableCodec foundation. The release therefore includes the StableCodec-derived backbone code required to construct and run the proposed model, while excluding StableCodec comparison experiments and cumulative ablation launchers.

HMCG-UIC transmits a main latent stream together with compact structure, color, and semantic side streams. The decoded structure and semantic evidence reconstructs the three-channel machine-oriented representation `F_hat_m`. Its `3->64->320` projection and the projected multi-prior condition guide one-step diffusion reconstruction, while the original four-channel prior residuals correct the scheduler output.

## Paper-aligned configuration

- Input resolution: `512 x 512`, with random horizontal and vertical flips.
- One-step SD-Turbo reconstruction at fixed timestep `t = 999`.
- Main synthesis feature and U-Net input width: 320 channels.
- Structure residual scale: `0.1`.
- Color and semantic codebooks: `256 x 64` each.
- Color and semantic payloads: four 8-bit indices on a `2 x 2` grid.
- Machine reconstruction: structure `1->64`, semantic `64->128`, fusion `192->128->64->3`, condition projection `3->64->320`.
- Multi-prior condition projection: `12->96->320`; condition injection leaves the first four scheduler-latent channels unchanged.
- Latent correction: `r = r_m + G_s + G_c + G_e`.
- Stage 1: 30,001 steps at `lambda_rate=0.5`, AdamW learning rate `1e-4`, betas `(0.9, 0.999)`, weight decay `1e-2`, gradient clipping `1.0`.
- Stage 2: 21,001 steps for `lambda_rate in {32, 24, 16, 12, 8, 6, 4}`.
- Stage-2 generator learning rate: `5e-5`, reduced to `2e-5`, `1e-5`, and `1e-6` at steps 5,000, 10,000, and 15,000. Discriminator learning rate: `2e-5`.
- Loss weights: `lambda_mse=2.0`, `lambda_lpips=1.0`, `lambda_clip=0.1`, `lambda_aux=1.0`, `lambda_structure=1.0`, `lambda_color=0.5`, `lambda_semantic=1.0`, `lambda_gan=0.1`, and `lambda_task=1.0`.
- Stage-2 task supervision uses frozen FathomNet YOLO and TC-USOD features on `F_hat_m`.
- Saliency evaluation uses an independent U2-Net model.
- Actual bpp is measured from the complete on-disk bitstream, including framing overhead.

## Repository layout

```text
configs/                    Machine-local path template
ELIC/                       Main learned entropy model
scripts/paper_common.sh     Shared manuscript configuration
src/hmcg_uic.py             Public HMCG-UIC model entry
src/stablecodec_backbone.py Retained StableCodec-derived reconstruction backbone
src/diffusion_utils.py      One-step scheduler and LoRA helpers used by the backbone
src/my_utils/vaehook.py     Tiled VAE execution used by the backbone
src/latent_codec.py         Main and three side-stream codecs
src/train.py                Stage-1 joint training
src/finetune.py             Stage-2 task-aware fine-tuning
src/compress.py             Compression and reconstruction
src/evaluate_*.py           Four manuscript evaluations
THIRD_PARTY_NOTICES.md      Inherited-code and attribution inventory
train.sh                    Reproduction entry for Stage 1
finetune.sh                 Reproduction entry for Stage 2
```

`HMCGUIC` subclasses `StableCodecBackbone`. The backbone retains the upstream SD-Turbo/LoRA/tiled-VAE/checkpoint machinery and instantiates the complete HMCG-UIC latent codec. This separation records the implementation lineage without exposing the StableCodec baseline as the proposed method.

Datasets, weights, generated outputs, result aggregation, plotting, traditional-codec runners, comparison-method launchers, and ablation-only launchers are not included. See `THIRD_PARTY_NOTICES.md` before public redistribution.

## Setup

Use Python 3.10 and install the dependencies:

```bash
pip install -r requirements.txt
cp configs/paper.env.example configs/paper.env
source configs/paper.env
```

Set the local paths in `configs/paper.env` for SD-Turbo, ELIC, the training data, FathomNet YOLO, TC-USOD, DINOv2, and U2-Net. External assets are not redistributed.

## Training

Stage 1 jointly trains the complete main, structure, color, semantic, machine-reconstruction, and diffusion-conditioning paths:

```bash
bash train.sh
```

The final Stage-1 checkpoint is:

```text
hmcg_uic_stage1_30001.pkl
```

Stage 2 fine-tunes all seven manuscript rate points:

```bash
bash finetune.sh
```

For example, the final checkpoint at `lambda_rate=4` is:

```text
hmcg_uic_lambda_4_stage2_21001.pkl
```

## Compression and reconstruction

```bash
python src/compress.py \
  --sd_path "$SD_PATH" \
  --elic_path "$ELIC_PATH" \
  --codec_path /path/to/hmcg_uic_lambda_4_stage2_21001.pkl \
  --img_path /path/to/images \
  --rec_path outputs/reconstruction \
  --machine_path outputs/machine_feature \
  --bin_path outputs/bitstreams \
  --stream_bpp_csv outputs/stream_bpp.csv \
  --lambda_rate 4 \
  --seed 2026
```

`rec_path` contains human-oriented reconstructions. `machine_path` contains the decoded machine-oriented representation used by downstream tasks.

## Evaluation

- `src/evaluate_fgo_quality.py`: foreground PSNR, MS-SSIM, DISTS, and LPIPS.
- `src/evaluate_u2net_saliency.py`: U2-Net S-measure, max F-measure, and MAE.
- `src/evaluate_suim_segmentation_keras.py`: SUIM mIoU.
- `src/evaluate_yolo_detection.py`: DUO mAP50.


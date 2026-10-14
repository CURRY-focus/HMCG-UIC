# Inherited-code inventory

HMCG-UIC is not implemented from an empty framework. The following retained
components are required to reproduce the proposed method and must remain in the
source release.

## StableCodec-derived foundation

- `src/stablecodec_backbone.py`: SD-Turbo loading, one-step reconstruction,
  VAE/U-Net LoRA setup, prompt conditioning, tiled inference, checkpoint I/O,
  auxiliary ELIC feature extraction, and the common forward/compress/decompress
  control flow. It has been adapted to instantiate the complete HMCG-UIC codec.
- `src/diffusion_utils.py`: one-step scheduler and LoRA forward helpers used by
  the retained backbone.
- `src/my_utils/vaehook.py`: tiled VAE execution used by the retained backbone.

These files are implementation dependencies of HMCG-UIC. They are not a
released StableCodec comparison-method experiment, and there is no separate
StableCodec training launcher in this repository.

## Other retained foundations

- `ELIC/`: the auxiliary learned compression analysis transform required by the
  model.
- `src/vision_aided_loss/`: the feature-discriminator implementation used by
  Stage-2 adversarial training. Individual source headers, including the
  `blurpool.py` CC BY-NC-SA notice, remain applicable.
- `src/my_utils/vaehook.py` contains its own tiled-VAE attribution and license
  header; those notices must be retained.

## Before a public GitHub release

This inventory is not a replacement for upstream license texts. Add the exact
StableCodec and ELIC repository URLs, pinned commits, and their license files
from the copies actually used for the experiments. Do not assign a new
repository-wide license until compatibility with every retained component has
been checked.

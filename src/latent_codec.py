import math
import struct
import sys
from pathlib import Path
from torch import Tensor
from typing import NamedTuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import quantize_ste as ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder
from ELIC.model.elic_official import CompressionModel, get_scale_table


class InceptionDWConv2d(nn.Module):
    def __init__(self, split_indexes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()

        self.dwconv_hw = nn.Conv2d(split_indexes[1], split_indexes[1], square_kernel_size,
                                   padding=square_kernel_size // 2, groups=split_indexes[1])
        self.dwconv_w = nn.Conv2d(split_indexes[2], split_indexes[2], kernel_size=(1, band_kernel_size),
                                  padding=(0, band_kernel_size // 2), groups=split_indexes[2])
        self.dwconv_h = nn.Conv2d(split_indexes[3], split_indexes[3], kernel_size=(band_kernel_size, 1),
                                  padding=(band_kernel_size // 2, 0), groups=split_indexes[3])
        self.split_indexes = split_indexes

    def forward(self, x):
        id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)


class InceptionNeXt(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.depthconv = InceptionDWConv2d((in_ch - (in_ch // 8) * 3, in_ch // 8, in_ch // 8, in_ch // 8))
        self.conv1 = nn.Conv2d(in_ch, in_ch * 2, 1)
        self.conv2 = nn.Conv2d(in_ch * 2, in_ch, 1)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.depthconv(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + shortcut


class GatedCNNBlock(nn.Module):
    def __init__(self, in_ch, expansion_ratio=2):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch, eps=1e-6)
        hidden = int(expansion_ratio * in_ch)
        self.fc1 = nn.Conv2d(in_ch, hidden * 2, 1)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.fc2 = nn.Conv2d(hidden, in_ch, 1)

    def forward(self, x):
        shortcut = x
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x1, x2 = self.fc1(x).chunk(2, 1)
        x = self.fc2(self.act(x1) * self.conv(x2))
        return x + shortcut


class BasicBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.blocks = nn.Sequential(
            InceptionNeXt(in_ch),
            GatedCNNBlock(in_ch),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=2, padding=2, groups=out_ch),
        )

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, padding=0),
            nn.PixelShuffle(2),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, padding=0),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)


class AnalysisTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre1 = Downsample(4, 128)
        self.pre2 = nn.Conv2d(320, 64, kernel_size=3, padding=1)
        self.analysis_transform = nn.Sequential(
            BasicBlock(192),
            Downsample(192, 256),
            BasicBlock(256),
            Downsample(256, 320),
            BasicBlock(320),
        )

    def forward(self, latent, latent2):
        x = torch.cat((self.pre1(latent), self.pre2(latent2)), dim=1)
        x = self.analysis_transform(x)
        return x


class SynthesisTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            BasicBlock(320),
            Upsample(320, 320),
            BasicBlock(320),
            Upsample(320, 320),
            BasicBlock(320),
            Upsample(320, 320),
        )

    def forward(self, x):
        x = self.synthesis_transform(x)
        return x


class AuxDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            BasicBlock(320),
            Upsample(320, 256),
            BasicBlock(256),
            Upsample(256, 192),
            BasicBlock(192),
            Upsample(192, 4),
        )

    def forward(self, x):
        x = self.block(x)
        return x


def extract_structure_map(image, eps=1e-6):
    """Build the normalized Sobel structure target defined in the paper.

    ``image`` is an RGB tensor in [-1, 1].  Keeping this transform in the
    codec module makes training-time forward coding and real entropy coding
    use exactly the same structural source.
    """
    image = ((image.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    sobel_x = gray.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    sobel_y = gray.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    edge = torch.sqrt(grad_x.square() + grad_y.square() + eps)
    scale = edge.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(eps)
    return edge / scale


class StructurePriorBranch(nn.Module):
    """Independent entropy-coded structure path ``S -> b_s -> S_hat, G_s``."""

    _STREAM_HEADER = ">4sHH"

    def __init__(self, residual_scale=0.1) -> None:
        super().__init__()
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))
        # The stride-8 stem first aligns the image-space sketch to the VAE
        # grid.  Three further reductions form the H/64 x W/64 payload.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=8, padding=3),
            BasicBlock(64),
            Downsample(64, 96),
            BasicBlock(96),
            Downsample(96, 192),
            BasicBlock(192),
            Downsample(192, 320),
            BasicBlock(320),
        )
        self.entropy_bottleneck = EntropyBottleneck(320)
        self.trunk = nn.Sequential(
            BasicBlock(320),
            Upsample(320, 192),
            BasicBlock(192),
            Upsample(192, 96),
            BasicBlock(96),
            Upsample(96, 64),
            BasicBlock(64),
        )
        self.sketch_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.attn_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 4, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 4, kernel_size=1),
        )

    def _decode(self, y_hat, out_size):
        feat = self.trunk(y_hat)
        if feat.shape[-2:] != out_size:
            feat = F.interpolate(feat, size=out_size, mode="bilinear", align_corners=False)
        sketch = self.sketch_head(feat)
        attn = self.attn_head(feat)
        residual = self.residual_scale.to(dtype=feat.dtype) * torch.tanh(self.residual_head(feat)) * attn
        return {
            "sketch": sketch,
            "attention": attn,
            "residual": residual,
            "latent": y_hat,
        }

    def forward(self, structure, out_size, ori_h, ori_w):
        y = self.encoder(structure)
        y_hat, likelihoods = self.entropy_bottleneck(y)
        info = self._decode(y_hat, out_size=out_size)
        bpp = -torch.log2(likelihoods.clamp_min(1e-9)).flatten(1).sum(dim=1)
        info["bpp_per_sample"] = bpp / float(ori_h * ori_w)
        info["bpp"] = info["bpp_per_sample"].detach().mean()
        return info

    @classmethod
    def _pack_payload(cls, payload, shape):
        height, width = int(shape[0]), int(shape[1])
        if height > 65535 or width > 65535:
            raise ValueError("Structure latent is too large to pack.")
        return struct.pack(cls._STREAM_HEADER, b"STRB", height, width) + payload

    @classmethod
    def _unpack_payload(cls, packed):
        header_size = struct.calcsize(cls._STREAM_HEADER)
        if len(packed) < header_size:
            raise ValueError("Truncated structure bitstream.")
        magic, height, width = struct.unpack(cls._STREAM_HEADER, packed[:header_size])
        if magic != b"STRB":
            raise ValueError("Invalid structure bitstream.")
        return packed[header_size:], (height, width)

    def compress(self, structure):
        y = self.encoder(structure)
        payloads = self.entropy_bottleneck.compress(y)
        return [self._pack_payload(payload, y.shape[-2:]) for payload in payloads]

    def decompress(self, packed_stream, out_size):
        payloads = []
        shape = None
        for packed in packed_stream:
            payload, current_shape = self._unpack_payload(packed)
            if shape is not None and current_shape != shape:
                raise ValueError("All structure samples must have the same latent shape.")
            shape = current_shape
            payloads.append(payload)
        if shape is None:
            raise ValueError("Empty structure bitstream.")
        y_hat = self.entropy_bottleneck.decompress(payloads, shape)
        info = self._decode(y_hat, out_size=out_size)
        info["bpp"] = y_hat.new_tensor(0.0)
        info["bpp_per_sample"] = y_hat.new_zeros((y_hat.shape[0],))
        return info


class MachineFeatureReconstructor(nn.Module):
    """Fuse decoded structure and semantic tokens into machine tensor F_hat_m.

    The three-channel output is a task-oriented tensor in [-1, 1], so fixed
    image task networks can consume it without changing their first layer.  It
    is not used as the human RGB reconstruction.  A separate 320-channel view
    is injected into the diffusion U-Net.
    """

    def __init__(self, semantic_dim=64):
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        # Paper Sec. II-D: 1->64 structure projection, 64->128 semantic
        # projection, and a 192->128 fusion convolution.
        self.structure_proj = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.semantic_proj = nn.Conv2d(semantic_dim, 128, kernel_size=1)
        self.fusion = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        self.feature_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 3, kernel_size=1),
            # F_hat is consumed by frozen image-task networks whose adapters
            # expect the same [-1, 1] domain as the source image.
            nn.Tanh(),
        )
        self.guidance_head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 320, kernel_size=1),
        )

    def forward(self, sketch, semantic_tokens, out_size):
        sketch = F.interpolate(sketch, size=out_size, mode="bilinear", align_corners=False)
        # Defensive fallback for isolated module tests. The complete HMCG-UIC
        # encode/decode path always supplies decoded semantic evidence here.
        if semantic_tokens is None:
            semantic_tokens = sketch.new_zeros(
                (sketch.shape[0], self.semantic_dim, 1, 1)
            )
        semantic = F.interpolate(semantic_tokens, size=out_size, mode="bilinear", align_corners=False)
        fused = self.fusion(torch.cat([self.structure_proj(sketch), self.semantic_proj(semantic)], dim=1))
        feature = self.feature_head(fused)
        return {
            "feature": feature,
            "guidance": self.guidance_head(feature),
        }


class ColorIndexBranch(nn.Module):
    def __init__(
            self,
            codebook_size=256,
            code_dim=64,
            residual_scale=0.05,
            commitment_beta=0.25,
            global_tokens=4,
    ) -> None:
        super().__init__()
        if codebook_size > 65536:
            raise ValueError("color_codebook_size must be <= 65536")
        if global_tokens < 0:
            raise ValueError("color_global_tokens must be >= 0")

        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_beta = commitment_beta
        self.global_tokens = int(global_tokens)
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))

        self.encoder = nn.Sequential(
            Downsample(4, 64),
            BasicBlock(64),
            Downsample(64, 64),
            BasicBlock(64),
            Downsample(64, code_dim),
            BasicBlock(code_dim),
        )
        self.codebook = nn.Embedding(codebook_size, code_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(code_dim, 64, kernel_size=1),
            BasicBlock(64),
            Upsample(64, 64),
            BasicBlock(64),
            Upsample(64, 64),
            BasicBlock(64),
            Upsample(64, 4),
        )
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def _global_grid(self):
        if self.global_tokens <= 0:
            return None
        height = int(math.floor(math.sqrt(self.global_tokens)))
        while height > 1 and self.global_tokens % height != 0:
            height -= 1
        width = self.global_tokens // height
        return height, width

    def _encode_tokens(self, latent):
        z_e = self.encoder(latent)
        grid = self._global_grid()
        if grid is not None:
            z_e = F.adaptive_avg_pool2d(z_e, grid)
        return z_e

    def _nearest_indices(self, z_e):
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.code_dim).float()
        codebook = self.codebook.weight.float()
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_flat @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = torch.argmin(dist, dim=1)
        return indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])

    def _lookup(self, indices, dtype=None):
        z_q = self.codebook(indices).permute(0, 3, 1, 2).contiguous()
        if dtype is not None:
            z_q = z_q.to(dtype=dtype)
        return z_q

    def _decode(self, z_q, out_size):
        residual = self.residual_scale.to(device=z_q.device, dtype=z_q.dtype) * torch.tanh(self.decoder(z_q))
        if residual.shape[-2:] != out_size:
            residual = F.interpolate(residual, size=out_size, mode="bilinear", align_corners=False)
        return residual

    def forward(self, latent, out_size):
        z_e = self._encode_tokens(latent)
        indices = self._nearest_indices(z_e)
        z_q = self._lookup(indices, dtype=z_e.dtype)
        z_q_st = z_e + (z_q - z_e).detach()

        codebook_loss = F.mse_loss(z_q.float(), z_e.detach().float())
        commitment_loss = F.mse_loss(z_e.float(), z_q.detach().float())
        vq_loss = codebook_loss + self.commitment_beta * commitment_loss

        residual = self._decode(z_q_st, out_size=out_size)
        return {
            "indices": indices,
            "residual": residual,
            "vq_loss": vq_loss,
            "bpp": None,
        }

    def encode_indices(self, latent):
        z_e = self._encode_tokens(latent)
        return self._nearest_indices(z_e)

    def decode_indices(self, indices, device, out_size):
        indices = indices.to(device=device, dtype=torch.long)
        z_q = self._lookup(indices)
        return {
            "indices": indices,
            "residual": self._decode(z_q, out_size=out_size),
            "vq_loss": z_q.new_tensor(0.0),
            "bpp": None,
        }

    def estimated_bpp(self, indices, ori_h, ori_w):
        bits_per_index = max(1, math.ceil(math.log2(self.codebook_size)))
        bits = indices[0].numel() * bits_per_index
        bpp = bits / float(ori_h * ori_w)
        return indices.new_full((indices.shape[0],), bpp, dtype=torch.float32)

    def pack_indices(self, indices):
        indices = indices.detach().cpu()
        if indices.dim() != 3:
            raise ValueError("Color indices must have shape [B, H, W].")
        if self.global_tokens > 0:
            if indices.shape[0] != 1:
                raise ValueError("Global color token packing supports batch size 1.")
            flat = indices.reshape(-1).tolist()
            if len(flat) != self.global_tokens:
                raise ValueError(f"Expected {self.global_tokens} global color tokens, got {len(flat)}.")
            if self.codebook_size <= 256:
                return bytes(int(value) for value in flat)
            return struct.pack(f">{len(flat)}H", *[int(value) for value in flat])

        batch, height, width = indices.shape
        if batch > 255 or height > 65535 or width > 65535:
            raise ValueError("Color index map is too large to pack.")

        bytes_per_index = 1 if self.codebook_size <= 256 else 2
        header = struct.pack(">4sBHHB", b"CLRB", batch, height, width, bytes_per_index)
        if bytes_per_index == 1:
            body = indices.to(torch.uint8).numpy().tobytes()
        else:
            body = indices.to(torch.int32).numpy().astype(">u2", copy=False).tobytes()
        return header + body

    def unpack_indices(self, packed, device):
        if self.global_tokens > 0:
            grid = self._global_grid()
            bytes_per_index = 1 if self.codebook_size <= 256 else 2
            expected_bytes = self.global_tokens * bytes_per_index
            if len(packed) != expected_bytes:
                raise ValueError(
                    f"Invalid global color token stream: expected {expected_bytes} bytes, got {len(packed)}."
                )
            if bytes_per_index == 1:
                values = torch.tensor(list(packed), dtype=torch.long, device=device)
            else:
                values = torch.tensor(
                    struct.unpack(f">{self.global_tokens}H", packed),
                    dtype=torch.long,
                    device=device,
                )
            return values.view(1, grid[0], grid[1])

        magic, batch, height, width, bytes_per_index = struct.unpack(">4sBHHB", packed[:10])
        if magic != b"CLRB":
            raise ValueError("Invalid color index stream.")

        body = packed[10:]
        n = batch * height * width
        if bytes_per_index == 1:
            values = torch.tensor(list(body), dtype=torch.long, device=device)
        elif bytes_per_index == 2:
            values = torch.tensor(struct.unpack(f">{n}H", body), dtype=torch.long, device=device)
        else:
            raise ValueError("Unsupported color index precision.")
        return values.view(batch, height, width)


class SemanticTokenBranch(nn.Module):
    def __init__(
            self,
            codebook_size=256,
            code_dim=64,
            residual_scale=0.05,
            commitment_beta=0.25,
            global_tokens=4,
    ) -> None:
        super().__init__()
        if codebook_size > 65536:
            raise ValueError("semantic_codebook_size must be <= 65536")
        if global_tokens < 0:
            raise ValueError("semantic_global_tokens must be >= 0")

        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_beta = commitment_beta
        self.global_tokens = int(global_tokens)
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))

        self.encoder = nn.Sequential(
            Downsample(4, 64),
            BasicBlock(64),
            Downsample(64, 96),
            BasicBlock(96),
            Downsample(96, code_dim),
            BasicBlock(code_dim),
        )
        self.codebook = nn.Embedding(codebook_size, code_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(code_dim, 96, kernel_size=1),
            BasicBlock(96),
            Upsample(96, 96),
            BasicBlock(96),
            Upsample(96, 64),
            BasicBlock(64),
            Upsample(64, 64),
            BasicBlock(64),
        )
        self.attn_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 4, kernel_size=1),
        )
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def _global_grid(self):
        if self.global_tokens <= 0:
            return None
        height = int(math.floor(math.sqrt(self.global_tokens)))
        while height > 1 and self.global_tokens % height != 0:
            height -= 1
        width = self.global_tokens // height
        return height, width

    def _encode_tokens(self, latent):
        z_e = self.encoder(latent)
        grid = self._global_grid()
        if grid is not None:
            z_e = F.adaptive_avg_pool2d(z_e, grid)
        return z_e

    def _nearest_indices(self, z_e):
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.code_dim).float()
        codebook = self.codebook.weight.float()
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_flat @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = torch.argmin(dist, dim=1)
        return indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])

    def _lookup(self, indices, dtype=None):
        z_q = self.codebook(indices).permute(0, 3, 1, 2).contiguous()
        if dtype is not None:
            z_q = z_q.to(dtype=dtype)
        return z_q

    def _decode(self, z_q, out_size):
        feat = self.decoder(z_q)
        if feat.shape[-2:] != out_size:
            feat = F.interpolate(feat, size=out_size, mode="bilinear", align_corners=False)
        attention = self.attn_head(feat)
        residual = (
            self.residual_scale.to(device=z_q.device, dtype=feat.dtype)
            * torch.tanh(self.residual_head(feat))
            * attention
        )
        return residual, attention, feat

    def forward(self, latent, out_size):
        z_e = self._encode_tokens(latent)
        indices = self._nearest_indices(z_e)
        z_q = self._lookup(indices, dtype=z_e.dtype)
        z_q_st = z_e + (z_q - z_e).detach()

        codebook_loss = F.mse_loss(z_q.float(), z_e.detach().float())
        commitment_loss = F.mse_loss(z_e.float(), z_q.detach().float())
        vq_loss = codebook_loss + self.commitment_beta * commitment_loss

        residual, attention, decoded_feature = self._decode(z_q_st, out_size=out_size)
        return {
            "indices": indices,
            "tokens": z_q_st,
            "decoded_feature": decoded_feature,
            "residual": residual,
            "attention": attention,
            "vq_loss": vq_loss,
            "bpp": None,
        }

    def encode_indices(self, latent):
        z_e = self._encode_tokens(latent)
        return self._nearest_indices(z_e)

    def decode_indices(self, indices, device, out_size):
        indices = indices.to(device=device, dtype=torch.long)
        z_q = self._lookup(indices)
        residual, attention, decoded_feature = self._decode(z_q, out_size=out_size)
        return {
            "indices": indices,
            "tokens": z_q,
            "decoded_feature": decoded_feature,
            "residual": residual,
            "attention": attention,
            "vq_loss": z_q.new_tensor(0.0),
            "bpp": None,
        }

    def estimated_bpp(self, indices, ori_h, ori_w):
        bits_per_index = max(1, math.ceil(math.log2(self.codebook_size)))
        bits = indices[0].numel() * bits_per_index
        bpp = bits / float(ori_h * ori_w)
        return indices.new_full((indices.shape[0],), bpp, dtype=torch.float32)

    def pack_indices(self, indices):
        indices = indices.detach().cpu()
        if indices.dim() != 3:
            raise ValueError("Semantic indices must have shape [B, H, W].")
        if self.global_tokens > 0:
            if indices.shape[0] != 1:
                raise ValueError("Global semantic token packing supports batch size 1.")
            flat = indices.reshape(-1).tolist()
            if len(flat) != self.global_tokens:
                raise ValueError(f"Expected {self.global_tokens} global semantic tokens, got {len(flat)}.")
            if self.codebook_size <= 256:
                return bytes(int(value) for value in flat)
            return struct.pack(f">{len(flat)}H", *[int(value) for value in flat])

        batch, height, width = indices.shape
        if batch > 255 or height > 65535 or width > 65535:
            raise ValueError("Semantic index map is too large to pack.")

        bytes_per_index = 1 if self.codebook_size <= 256 else 2
        header = struct.pack(">4sBHHB", b"SEMB", batch, height, width, bytes_per_index)
        if bytes_per_index == 1:
            body = indices.to(torch.uint8).numpy().tobytes()
        else:
            body = indices.to(torch.int32).numpy().astype(">u2", copy=False).tobytes()
        return header + body

    def unpack_indices(self, packed, device):
        if self.global_tokens > 0:
            grid = self._global_grid()
            bytes_per_index = 1 if self.codebook_size <= 256 else 2
            expected_bytes = self.global_tokens * bytes_per_index
            if len(packed) != expected_bytes:
                raise ValueError(
                    f"Invalid global semantic token stream: expected {expected_bytes} bytes, got {len(packed)}."
                )
            if bytes_per_index == 1:
                values = torch.tensor(list(packed), dtype=torch.long, device=device)
            else:
                values = torch.tensor(
                    struct.unpack(f">{self.global_tokens}H", packed),
                    dtype=torch.long,
                    device=device,
                )
            return values.view(1, grid[0], grid[1])

        magic, batch, height, width, bytes_per_index = struct.unpack(">4sBHHB", packed[:10])
        if magic != b"SEMB":
            raise ValueError("Invalid semantic index stream.")

        body = packed[10:]
        n = batch * height * width
        if bytes_per_index == 1:
            values = torch.tensor(list(body), dtype=torch.long, device=device)
        elif bytes_per_index == 2:
            values = torch.tensor(struct.unpack(f">{n}H", body), dtype=torch.long, device=device)
        else:
            raise ValueError("Unsupported semantic index precision.")
        return values.view(batch, height, width)


class Adapter(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, (in_ch + out_ch) // 2, 1),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, (in_ch + out_ch) // 2, 5, padding=2, groups=(in_ch + out_ch) // 2),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, out_ch, 1),
        )
        self.branch2 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)


class HyperAnalysis(nn.Module):
    def __init__(self, M=320) -> None:
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Conv2d(M, M // 2, 3, stride=2, padding=1),
            BasicBlock(M // 2),
            nn.Conv2d(M // 2, M // 2, 3, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class HyperSynthesis(nn.Module):
    def __init__(self, M=320) -> None:
        super().__init__()
        self.increase = nn.Sequential(
            nn.Conv2d(M // 2, M * 2, kernel_size=1, padding=0),
            nn.PixelShuffle(2),
            BasicBlock(M // 2),
            nn.Conv2d(M // 2, M * 4, kernel_size=1, padding=0),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        x = self.increase(x)
        return x


class SpatialContext(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block = nn.Sequential(
            BasicBlock(in_ch),
            BasicBlock(in_ch),
            BasicBlock(in_ch),
            BasicBlock(in_ch),
        )

    def forward(self, x):
        context = self.block(x)
        return context


class LRP(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Adapter(in_ch, (in_ch + out_ch) // 2),
            Adapter((in_ch + out_ch) // 2, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class LatentCodec(CompressionModel):
    def __init__(
            self,
            lambda_rate,
            structure_residual_scale=0.1,
            color_codebook_size=256,
            color_code_dim=64,
            color_residual_scale=0.05,
            color_global_tokens=4,
            semantic_codebook_size=256,
            semantic_code_dim=64,
            semantic_residual_scale=0.05,
            semantic_global_tokens=4,
    ):
        super().__init__()

        M = 320
        self.g_a = AnalysisTransform()
        self.g_s = SynthesisTransform()
        self.h_a = HyperAnalysis(M=M)
        self.h_s = HyperSynthesis(M=M)
        self.aux = AuxDecoder()
        self.structure = StructurePriorBranch(residual_scale=structure_residual_scale)
        self.color = ColorIndexBranch(
            codebook_size=color_codebook_size,
            code_dim=color_code_dim,
            residual_scale=color_residual_scale,
            global_tokens=color_global_tokens,
        )
        self.semantic = SemanticTokenBranch(
            codebook_size=semantic_codebook_size,
            code_dim=semantic_code_dim,
            residual_scale=semantic_residual_scale,
            global_tokens=semantic_global_tokens,
        )
        self.machine_reconstructor = MachineFeatureReconstructor(semantic_dim=semantic_code_dim)
        self.prior_conditioner = nn.Sequential(
            nn.Conv2d(12, 96, kernel_size=3, padding=1),
            nn.GELU(),
            BasicBlock(96),
            nn.Conv2d(96, M, kernel_size=1),
        )

        context_dim = M * 3
        self.adapter_in = nn.ModuleList(Adapter(in_ch=M, out_ch=context_dim) for i in range(4))
        self.g_c = SpatialContext(in_ch=context_dim)
        self.adapter_out = nn.ModuleList(Adapter(in_ch=context_dim, out_ch=M * 2) for i in range(4))
        self.LRP = nn.ModuleList(LRP(in_ch=M * 2, out_ch=M) for i in range(4))

        self.entropy_bottleneck = EntropyBottleneck(M // 2)
        self.gaussian_conditional = GaussianConditional(None)
        self.masks = {}

        self.rate = TargetRateModule(lambda_rate)

    @staticmethod
    def _without_scheduler_channels(condition):
        """Keep Pi_4(y_hat_m) unchanged while conditioning the 320-ch U-Net."""
        return torch.cat([torch.zeros_like(condition[:, :4]), condition[:, 4:]], dim=1)

    def _assemble_decoder_inputs(self, y_hat, structure_info, color_info, semantic_info):
        main_feature = self.g_s(y_hat)
        residual = self.aux(y_hat)

        side_residuals = []
        for info in (structure_info, color_info, semantic_info):
            if info is None:
                side_residuals.append(torch.zeros_like(residual))
            else:
                side_residuals.append(info["residual"])
                residual = residual + info["residual"]

        denoiser_input = main_feature
        if any(info is not None for info in (structure_info, color_info, semantic_info)):
            prior_condition = self.prior_conditioner(torch.cat(side_residuals, dim=1))
            denoiser_input = denoiser_input + self._without_scheduler_channels(prior_condition)

        machine_info = None
        if structure_info is not None:
            machine_info = self.machine_reconstructor(
                structure_info["sketch"],
                semantic_info["tokens"] if semantic_info is not None else None,
                out_size=main_feature.shape[-2:],
            )
            denoiser_input = denoiser_input + self._without_scheduler_channels(machine_info["guidance"])

        return denoiser_input, residual, machine_info

    def _add_side_rate(self, rate_output, side_bpp):
        return RateLossOutput(
            rate_loss=rate_output.rate_loss + (self.rate.lambda_rate * side_bpp).mean(),
            quantized_total_bpp=rate_output.quantized_total_bpp + side_bpp.detach().mean(),
            quantized_latent_bpp=rate_output.quantized_latent_bpp,
            quantized_hyper_bpp=rate_output.quantized_hyper_bpp,
        )

    def get_mask_four_parts(self, batch, channel, height, width, device='cuda'):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        if curr_mask_str not in self.masks:
            micro_m0 = torch.tensor(((1., 0), (0, 0)), device=device)
            m0 = micro_m0.repeat((height + 1) // 2, (width + 1) // 2)
            m0 = m0[:height, :width]
            m0 = torch.unsqueeze(m0, 0)
            m0 = torch.unsqueeze(m0, 0)

            micro_m1 = torch.tensor(((0, 1.), (0, 0)), device=device)
            m1 = micro_m1.repeat((height + 1) // 2, (width + 1) // 2)
            m1 = m1[:height, :width]
            m1 = torch.unsqueeze(m1, 0)
            m1 = torch.unsqueeze(m1, 0)

            micro_m2 = torch.tensor(((0, 0), (1., 0)), device=device)
            m2 = micro_m2.repeat((height + 1) // 2, (width + 1) // 2)
            m2 = m2[:height, :width]
            m2 = torch.unsqueeze(m2, 0)
            m2 = torch.unsqueeze(m2, 0)

            micro_m3 = torch.tensor(((0, 0), (0, 1.)), device=device)
            m3 = micro_m3.repeat((height + 1) // 2, (width + 1) // 2)
            m3 = m3[:height, :width]
            m3 = torch.unsqueeze(m3, 0)
            m3 = torch.unsqueeze(m3, 0)

            m = torch.ones((batch, channel // 4, height, width), device=device)
            mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
            mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
            mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
            mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)
            self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    def sequeeze_with_mask(self, latent, mask):
        latent_group_1, latent_group_2, latent_group_3, latent_group_4 = latent.chunk(4, 1)
        mask_group_1, mask_group_2, mask_group_3, mask_group_4 = mask.chunk(4, 1)
        latent_sequeeze = latent_group_1 * mask_group_1 + latent_group_2 * mask_group_2 + latent_group_3 * mask_group_3 + latent_group_4 * mask_group_4
        return latent_sequeeze

    def unsequeeze_with_mask(self, latent_sequeeze, mask):
        mask_group_1, mask_group_2, mask_group_3, mask_group_4 = mask.chunk(4, 1)
        latent = torch.cat((latent_sequeeze * mask_group_1, latent_sequeeze * mask_group_2,
                            latent_sequeeze * mask_group_3, latent_sequeeze * mask_group_4), dim=1)
        return latent

    def compress_group_with_mask(self, gaussian_conditional, latent, scales, means, mask, symbols_list, indexes_list):
        latent_squeeze = self.sequeeze_with_mask(latent, mask)
        scales_squeeze = self.sequeeze_with_mask(scales, mask)
        means_squeeze = self.sequeeze_with_mask(means, mask)
        indexes = gaussian_conditional.build_indexes(scales_squeeze)
        latent_squeeze_hat = gaussian_conditional.quantize(latent_squeeze, "symbols", means_squeeze)
        symbols_list.extend(latent_squeeze_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        latent_hat = self.unsequeeze_with_mask(latent_squeeze_hat + means_squeeze, mask)
        return latent_hat

    def decompress_group_with_mask(self, gaussian_conditional, scales, means, mask, decoder, cdf, cdf_lengths, offsets):
        scales_squeeze = self.sequeeze_with_mask(scales, mask)
        means_squeeze = self.sequeeze_with_mask(means, mask)
        indexes = gaussian_conditional.build_indexes(scales_squeeze)
        latent_squeeze_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        latent_squeeze_hat = torch.Tensor(latent_squeeze_hat).reshape(scales_squeeze.shape).to(scales.device)
        latent_hat = self.unsequeeze_with_mask(latent_squeeze_hat + means_squeeze, mask)
        return latent_hat

    def forward(self, latent, latent2, ori_h, ori_w, structure=None):

        y = self.g_a(latent, latent2)
        z = self.h_a(y)

        _, z_likelihoods = self.entropy_bottleneck(z)
        with torch.no_grad():
            _, quantized_z_likelihoods = self.entropy_bottleneck(z, training=False)

        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        B, C, H, W = y.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, device=y.device)

        # Hyper-parameters
        base = self.h_s(z_hat)
        means_0_supp, scales_0_supp = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        means_0 = means_0_supp * mask_0
        scales_0 = scales_0_supp * mask_0
        y_0 = y * mask_0
        y_hat_0 = ste_round(y_0 - means_0) + means_0
        lrp = self.LRP[0](torch.cat([y_hat_0, base], dim=1)) * mask_0
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_0 = y_hat_0 + lrp

        base = base * (1 - mask_0) + y_hat_0
        means_1_supp, scales_1_supp = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        means_1 = means_1_supp * mask_1
        scales_1 = scales_1_supp * mask_1
        y_1 = y * mask_1
        y_hat_1 = ste_round(y_1 - means_1) + means_1
        lrp = self.LRP[1](torch.cat([y_hat_1, base], dim=1)) * mask_1
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_1 = y_hat_1 + lrp

        base = base * (1 - mask_1) + y_hat_1
        means_2_supp, scales_2_supp = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        means_2 = means_2_supp * mask_2
        scales_2 = scales_2_supp * mask_2
        y_2 = y * mask_2
        y_hat_2 = ste_round(y_2 - means_2) + means_2
        lrp = self.LRP[2](torch.cat([y_hat_2, base], dim=1)) * mask_2
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_2 = y_hat_2 + lrp

        base = base * (1 - mask_2) + y_hat_2
        means_3_supp, scales_3_supp = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        means_3 = means_3_supp * mask_3
        scales_3 = scales_3_supp * mask_3
        y_3 = y * mask_3
        y_hat_3 = ste_round(y_3 - means_3) + means_3
        lrp = self.LRP[3](torch.cat([y_hat_3, base], dim=1)) * mask_3
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_3 = y_hat_3 + lrp

        scales_all = scales_0 + scales_1 + scales_2 + scales_3
        means_all = means_0 + means_1 + means_2 + means_3

        _, y_likelihoods = self.gaussian_conditional(y, scales_all, means_all)
        with torch.no_grad():
            _, quantized_y_likelihoods = self.gaussian_conditional(y, scales_all, means_all, training=False)

        y_hat = base * (1 - mask_3) + y_hat_3
        decoder_size = (y_hat.shape[-2] * 8, y_hat.shape[-1] * 8)
        if structure is None:
            raise ValueError("HMCG-UIC requires the structure map S=E_edge(x).")
        structure_info = self.structure(
            structure,
            out_size=decoder_size,
            ori_h=ori_h,
            ori_w=ori_w,
        )
        color_info = self.color(latent, out_size=decoder_size)
        semantic_info = self.semantic(latent, out_size=decoder_size)

        x_hat, res, machine_info = self._assemble_decoder_inputs(
            y_hat,
            structure_info,
            color_info,
            semantic_info,
        )

        RateLossOutput = self.rate(
            latent_likelihoods=y_likelihoods,
            quantized_latent_likelihoods=quantized_y_likelihoods,
            hyper_latent_likelihoods=z_likelihoods,
            quantized_hyper_latent_likelihoods=quantized_z_likelihoods,
            ori_h=ori_h,
            ori_w=ori_w,
        )
        RateLossOutput = self._add_side_rate(
            RateLossOutput,
            structure_info["bpp_per_sample"],
        )
        color_bpp = self.color.estimated_bpp(color_info["indices"], ori_h=ori_h, ori_w=ori_w)
        color_info["bpp"] = color_bpp.detach().mean()
        RateLossOutput = self._add_side_rate(RateLossOutput, color_bpp)
        semantic_bpp = self.semantic.estimated_bpp(semantic_info["indices"], ori_h=ori_h, ori_w=ori_w)
        semantic_info["bpp"] = semantic_bpp.detach().mean()
        RateLossOutput = self._add_side_rate(RateLossOutput, semantic_bpp)

        return x_hat, RateLossOutput, res, structure_info, color_info, semantic_info, machine_info

    def compress(self, latent, latent2, structure=None):

        if latent.shape[0] != 1:
            raise ValueError("File bitstream coding currently supports batch size 1.")

        y = self.g_a(latent, latent2)
        z = self.h_a(y)

        torch.backends.cudnn.deterministic = True
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        B, C, H, W = y.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, device=y.device)

        base = self.h_s(z_hat)
        means_0_supp, scales_0_supp = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        y_hat_0 = self.compress_group_with_mask(self.gaussian_conditional, y, scales_0_supp, means_0_supp, mask_0,
                                                symbols_list, indexes_list)
        lrp = self.LRP[0](torch.cat([y_hat_0, base], dim=1)) * mask_0
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_0 = y_hat_0 + lrp

        base = base * (1 - mask_0) + y_hat_0
        means_1_supp, scales_1_supp = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        y_hat_1 = self.compress_group_with_mask(self.gaussian_conditional, y, scales_1_supp, means_1_supp, mask_1,
                                                symbols_list, indexes_list)
        lrp = self.LRP[1](torch.cat([y_hat_1, base], dim=1)) * mask_1
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_1 = y_hat_1 + lrp

        base = base * (1 - mask_1) + y_hat_1
        means_2_supp, scales_2_supp = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        y_hat_2 = self.compress_group_with_mask(self.gaussian_conditional, y, scales_2_supp, means_2_supp, mask_2,
                                                symbols_list, indexes_list)
        lrp = self.LRP[2](torch.cat([y_hat_2, base], dim=1)) * mask_2
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_2 = y_hat_2 + lrp

        base = base * (1 - mask_2) + y_hat_2
        means_3_supp, scales_3_supp = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        _ = self.compress_group_with_mask(self.gaussian_conditional, y, scales_3_supp, means_3_supp, mask_3,
                                          symbols_list, indexes_list)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        torch.backends.cudnn.deterministic = False
        strings = [y_strings, z_strings]
        stream_names = ["main_latent", "main_hyperprior"]
        if structure is None:
            raise ValueError("HMCG-UIC requires the structure map S=E_edge(x).")
        strings.append(self.structure.compress(structure))
        stream_names.append("structure")
        color_indices = self.color.encode_indices(latent)
        strings.append([self.color.pack_indices(color_indices)])
        stream_names.append("color")
        semantic_indices = self.semantic.encode_indices(latent)
        strings.append([self.semantic.pack_indices(semantic_indices)])
        stream_names.append("semantic")

        return {
            "strings": strings,
            "shape": z.size()[-2:],
            "stream_names": stream_names,
        }

    def decompress(self, strings, shape):

        expected_streams = 5
        if len(strings) != expected_streams:
            raise ValueError(
                f"Bitstream/model mismatch: HMCG-UIC expects five streams "
                f"(two main, structure, color, semantic), got {len(strings)}."
            )

        torch.backends.cudnn.deterministic = True
        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        B, C, H, W = z_hat.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C * 2, H * 4, W * 4, device=z_hat.device)

        base = self.h_s(z_hat)
        means_0_supp, scales_0_supp = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        y_hat_0 = self.decompress_group_with_mask(self.gaussian_conditional, scales_0_supp, means_0_supp, mask_0,
                                                  decoder, cdf, cdf_lengths, offsets)
        lrp = self.LRP[0](torch.cat([y_hat_0, base], dim=1)) * mask_0
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_0 = y_hat_0 + lrp

        base = base * (1 - mask_0) + y_hat_0
        means_1_supp, scales_1_supp = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        y_hat_1 = self.decompress_group_with_mask(self.gaussian_conditional, scales_1_supp, means_1_supp, mask_1,
                                                  decoder, cdf, cdf_lengths, offsets)
        lrp = self.LRP[1](torch.cat([y_hat_1, base], dim=1)) * mask_1
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_1 = y_hat_1 + lrp

        base = base * (1 - mask_1) + y_hat_1
        means_2_supp, scales_2_supp = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        y_hat_2 = self.decompress_group_with_mask(self.gaussian_conditional, scales_2_supp, means_2_supp, mask_2,
                                                  decoder, cdf, cdf_lengths, offsets)
        lrp = self.LRP[2](torch.cat([y_hat_2, base], dim=1)) * mask_2
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_2 = y_hat_2 + lrp

        base = base * (1 - mask_2) + y_hat_2
        means_3_supp, scales_3_supp = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        y_hat_3 = self.decompress_group_with_mask(self.gaussian_conditional, scales_3_supp, means_3_supp, mask_3,
                                                  decoder, cdf, cdf_lengths, offsets)
        lrp = self.LRP[3](torch.cat([y_hat_3, base], dim=1)) * mask_3
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_3 = y_hat_3 + lrp

        y_hat = y_hat_0 + y_hat_1 + y_hat_2 + y_hat_3
        torch.backends.cudnn.deterministic = False

        decoder_size = (y_hat.shape[-2] * 8, y_hat.shape[-1] * 8)
        structure_info = self.structure.decompress(strings[2], out_size=decoder_size)
        color_indices = self.color.unpack_indices(strings[3][0], device=y_hat.device)
        color_info = self.color.decode_indices(color_indices, device=y_hat.device, out_size=decoder_size)
        semantic_indices = self.semantic.unpack_indices(strings[4][0], device=y_hat.device)
        semantic_info = self.semantic.decode_indices(
            semantic_indices,
            device=y_hat.device,
            out_size=decoder_size,
        )

        x_hat, res, machine_info = self._assemble_decoder_inputs(
            y_hat,
            structure_info,
            color_info,
            semantic_info,
        )

        return x_hat, res, structure_info, color_info, semantic_info, machine_info

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        updated |= self.structure.entropy_bottleneck.update(force=force)
        return updated


class RateLossOutput(NamedTuple):
    rate_loss: Tensor
    quantized_total_bpp: Tensor
    quantized_latent_bpp: Tensor
    quantized_hyper_bpp: Tensor


class TargetRateModule(nn.Module):
    def __init__(self, lambda_rate):
        super().__init__()
        self.lambda_rate = lambda_rate

    def _calc_bits_per_batch(self, likelihoods: Tensor) -> Tensor:
        batch_size = likelihoods.shape[0]
        likelihoods = likelihoods.reshape(batch_size, -1)
        return likelihoods.log().sum(1) / -math.log(2)

    def forward(
            self,
            latent_likelihoods: Tensor,
            quantized_latent_likelihoods: Tensor,
            hyper_latent_likelihoods: Tensor,
            quantized_hyper_latent_likelihoods: Tensor,
            ori_h=None,
            ori_w=None,
    ):
        num_pixels = ori_h * ori_w

        latent_bpp = self._calc_bits_per_batch(latent_likelihoods) / num_pixels
        quantized_latent_bpp = self._calc_bits_per_batch(quantized_latent_likelihoods) / num_pixels
        hyper_bpp = self._calc_bits_per_batch(hyper_latent_likelihoods) / num_pixels
        quantized_hyper_bpp = self._calc_bits_per_batch(quantized_hyper_latent_likelihoods) / num_pixels

        total_bpp = latent_bpp + hyper_bpp
        quantized_total_bpp = quantized_latent_bpp + quantized_hyper_bpp

        return RateLossOutput(
            rate_loss=(self.lambda_rate * total_bpp).mean(),
            quantized_total_bpp=quantized_total_bpp.detach().mean(),
            quantized_latent_bpp=quantized_latent_bpp.detach().mean(),
            quantized_hyper_bpp=quantized_hyper_bpp.detach().mean(),
        )

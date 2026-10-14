"""StableCodec-derived backbone used by HMCG-UIC.

This module preserves the SD-Turbo, LoRA, tiled-VAE, prompt-conditioning,
auxiliary ELIC, checkpoint, and entropy-coding foundation inherited from
StableCodec.  It is not a separate comparison-method launcher: the codec
instantiated below is the complete HMCG-UIC codec defined in ``latent_codec``.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from diffusion_utils import make_1step_sched, my_lora_fwd
from my_utils.vaehook import VAEHook
from latent_codec import LatentCodec, extract_structure_map
from ELIC.model.elic_official import ELIC

class StableCodecBackbone(torch.nn.Module):
    """Adapted StableCodec foundation with the complete HMCG-UIC codec."""

    def __init__(self, sd_path=None, args=None):
        super().__init__()

        self.latent_tiled_size = args.latent_tiled_size
        self.latent_tiled_overlap = args.latent_tiled_overlap

        print("[SD-Turbo]: Building SD-Turbo ......")
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(sd_path)
        self.guidance_scale = 1.07

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

        self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)
        print("[SD-Turbo]: Done!")

        print("[LoRA]: Initializing LoRA ......")
        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]
        lora_rank_vae = args.lora_rank_vae
        lora_rank_unet = args.lora_rank_unet

        vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian", target_modules=target_modules_vae)
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet)
        self.unet.add_adapter(unet_lora_config)

        self.vae_lora_layers = []
        for name, module in self.vae.named_modules():
            if 'base_layer' in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
        for name, module in self.vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in self.unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])
        for name, module in self.unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)
        print("[LoRA]: Done!")

        print("[Latent Codec]: Initializing Latent Codec ......")
        self.codec = LatentCodec(
            args.lambda_rate,
            structure_residual_scale=getattr(args, "structure_residual_scale", 0.1),
            color_codebook_size=getattr(args, "color_codebook_size", 256),
            color_code_dim=getattr(args, "color_code_dim", 64),
            color_residual_scale=getattr(args, "color_residual_scale", 0.05),
            color_global_tokens=getattr(args, "color_global_tokens", 4),
            semantic_codebook_size=getattr(args, "semantic_codebook_size", 256),
            semantic_code_dim=getattr(args, "semantic_code_dim", 64),
            semantic_residual_scale=getattr(args, "semantic_residual_scale", 0.05),
            semantic_global_tokens=getattr(args, "semantic_global_tokens", 4),
        )
        temp_layer = nn.Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.unet.conv_in = temp_layer
        print("[Latent Codec]: Done!")

        print("[Prompt]: Setting Prompt ......")
        self.set_prompt(args.pos_prompt)
        del self.tokenizer, self.text_encoder
        print("[Prompt]: Done!")

        if args.codec_path is not None:
            print("[LoRA & Latent Codec & Auxiliary Decoder]: Loading Pretrained Weights ......")
            sd = torch.load(args.codec_path, map_location="cpu")
            codec_checkpoint = sd["state_dict_codec"]
            required_prefixes = [
                "g_a.", "g_s.", "h_a.", "h_s.", "aux.",
                "adapter_in.", "g_c.", "adapter_out.", "LRP.",
                "entropy_bottleneck.", "gaussian_conditional.",
            ]
            required_prefixes.extend(
                [
                    "structure.",
                    "color.",
                    "semantic.",
                    "machine_reconstructor.",
                    "prior_conditioner.",
                ]
            )

            current_codec = self.codec.state_dict()
            required_keys = {
                key
                for key in current_codec
                if any(key.startswith(prefix) for prefix in required_prefixes)
            }
            missing_keys = sorted(required_keys.difference(codec_checkpoint))
            shape_mismatches = sorted(
                key
                for key in required_keys.intersection(codec_checkpoint)
                if tuple(current_codec[key].shape) != tuple(codec_checkpoint[key].shape)
            )
            if missing_keys or shape_mismatches:
                details = []
                if missing_keys:
                    details.append("missing=" + ", ".join(missing_keys[:12]))
                if shape_mismatches:
                    details.append("shape-mismatch=" + ", ".join(shape_mismatches[:12]))
                message = (
                    "Checkpoint is incomplete for the active paper architecture: "
                    + "; ".join(details)
                    + ". Retrain/fine-tune this architecture before reporting results."
                )
                if getattr(args, "require_paper_checkpoint", False):
                    raise RuntimeError(message)
                print(f"[Checkpoint migration warning] {message}")

            _sd_codec = current_codec
            for k, v in codec_checkpoint.items():
                if k in _sd_codec and tuple(_sd_codec[k].shape) == tuple(v.shape):
                    _sd_codec[k] = v
            self.codec.load_state_dict(_sd_codec)

            _sd_vae = self.vae.state_dict()
            required_vae_keys = {key for key in _sd_vae if "lora" in key}
            missing_vae_keys = sorted(required_vae_keys.difference(sd["state_dict_vae"]))
            if missing_vae_keys and getattr(args, "require_paper_checkpoint", False):
                raise RuntimeError(
                    "Checkpoint is missing VAE LoRA tensors: " + ", ".join(missing_vae_keys[:12])
                )
            for k, v in sd["state_dict_vae"].items():
                if k in _sd_vae and tuple(_sd_vae[k].shape) == tuple(v.shape):
                    _sd_vae[k] = v
            self.vae.load_state_dict(_sd_vae)

            _sd_unet = self.unet.state_dict()
            required_unet_keys = {
                key for key in _sd_unet if "lora" in key or key.startswith("conv_in.")
            }
            missing_unet_keys = sorted(required_unet_keys.difference(sd["state_dict_unet"]))
            if missing_unet_keys and getattr(args, "require_paper_checkpoint", False):
                raise RuntimeError(
                    "Checkpoint is missing U-Net LoRA/conv_in tensors: "
                    + ", ".join(missing_unet_keys[:12])
                )
            for k, v in sd["state_dict_unet"].items():
                if k in _sd_unet and tuple(_sd_unet[k].shape) == tuple(v.shape):
                    _sd_unet[k] = v
            self.unet.load_state_dict(_sd_unet)
            print("[LoRA & Latent Codec & Auxiliary Decoder]: Done!")

        print("[Auxiliary Encoder]: Loading Pretrained Weights ......")
        model = ELIC()
        checkpoint = torch.load(args.elic_path)
        model.load_state_dict(checkpoint)
        self.aux_codec = model.g_a
        self.aux_codec.eval()
        self.aux_codec.requires_grad_(False)
        print("[Auxiliary Encoder]: Done!")

    def set_prompt(self, pos_prompt):
        caption_tokens = self.tokenizer(pos_prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
        self.pos_caption_enc = self.text_encoder(caption_tokens)[0]

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.codec.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.codec.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.codec.train()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.codec.requires_grad_(True)

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def forward(self, x, pos_prompt, ori_h, ori_w, return_aux=False):
        # Encoder
        with torch.no_grad():
            latent2 = self.aux_codec((x + 1) / 2).detach()
            pos_caption_enc = [self.pos_caption_enc for i in range(len(pos_prompt))]
            pos_caption_enc = torch.cat(pos_caption_enc, dim=0).to(x.device)
        lq_latent = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor
        structure = extract_structure_map(x)

        # Latent Codec
        lq_latent_hat, RateLossOutput, res1, structure_info, color_info, semantic_info, machine_info = self.codec(
            lq_latent,
            latent2,
            ori_h,
            ori_w,
            structure=structure,
        )

        # One-Step Denoiser
        model_pred = self.unet(lq_latent_hat, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, lq_latent_hat[:, :4], return_dict=True).prev_sample + res1

        # Decoder
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        if return_aux:
            if machine_info is not None:
                machine_info = dict(machine_info)
                machine_info["feature_latent"] = machine_info["feature"]
                machine_info["feature"] = torch.nn.functional.interpolate(
                    machine_info["feature"],
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(-1, 1)
            return output_image, RateLossOutput, {
                "structure": structure_info,
                "color": color_info,
                "semantic": semantic_info,
                "machine": machine_info,
            }
        return output_image, RateLossOutput

    def compress(self, x):
        # Encoder
        latent2 = self.aux_codec((x + 1) / 2).detach()
        lq_latent = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor
        structure = extract_structure_map(x)

        # Latent Codec - Entropy Encoding
        output_dict = self.codec.compress(lq_latent, latent2, structure=structure)

        return output_dict

    def decompress(self, strings, shape, pos_prompt, return_aux=False):
        # Latent Codec - Entropy Decoding
        lq_latent_hat, res, structure_info, color_info, semantic_info, machine_info = self.codec.decompress(
            strings,
            shape,
        )

        pos_caption_enc = [self.pos_caption_enc for i in range(len(pos_prompt))]
        pos_caption_enc = torch.cat(pos_caption_enc, dim=0).to(lq_latent_hat.device)

        # One-Step Denoiser with tile function
        _, _, h, w = lq_latent_hat.size()
        tile_size, tile_overlap = (self.latent_tiled_size, self.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            model_pred = self.unet(lq_latent_hat, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
        else:
            print(f"[Tiled Latent]: the input latent is {h}x{w}, need to tiled")
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to(lq_latent_hat.device)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent_hat.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent_hat.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    input_tile = lq_latent_hat[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        model_pred = self.unet(input_list_t, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
                        input_list = []
                    noise_preds.append(model_pred)

            noise_pred = torch.zeros(lq_latent_hat[:, :4].shape, device=lq_latent_hat.device)
            contributors = torch.zeros(lq_latent_hat[:, :4].shape, device=lq_latent_hat.device)
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            noise_pred /= contributors
            model_pred = noise_pred

        x_denoised = self.sched.step(model_pred, self.timesteps, lq_latent_hat[:, :4], return_dict=True).prev_sample + res

        # Decoder
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        if return_aux:
            if machine_info is not None:
                machine_info = dict(machine_info)
                machine_info["feature_latent"] = machine_info["feature"]
                machine_info["feature"] = torch.nn.functional.interpolate(
                    machine_info["feature"],
                    size=output_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(-1, 1)
            return output_image, {
                "structure": structure_info,
                "color": color_info,
                "semantic": semantic_info,
                "machine": machine_info,
            }
        return output_image

    def save_model(self, outf):
        sd = {}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_codec"] = {k: v for k, v in self.codec.state_dict().items()}
        torch.save(sd, outf)

    def _set_latent_tile(self, latent_tiled_size = 96, latent_tiled_overlap = 32):
        self.latent_tiled_size = latent_tiled_size
        self.latent_tiled_overlap = latent_tiled_overlap

    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            vae_to_gpu = True):
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights), (nbatches, self.unet.config.in_channels, 1, 1))

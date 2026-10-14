import os
import sys
import gc
import lpips
import numpy as np
import transformers
import torch
import torch.utils.checkpoint
import diffusers

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.abspath(os.path.dirname(__file__))
for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from my_utils.training_utils import (
    parse_args_training,
    H5Dataset,
    ImageDepthFolder,
    ImageDepthTrainTransform,
    ImageDepthEvalTransform,
    CLIPLoss,
)
from my_utils.structure_utils import StructureLoss
from my_utils.color_utils import ColorLoss
from my_utils.semantic_utils import SemanticLoss
from my_utils.task_utils import MachineTaskLoss
from hmcg_uic import HMCGUIC
import vision_aided_loss


def write_eval_log(output_dir, global_step, logs, progress_bar):
    log_items = []
    for key, value in logs.items():
        if hasattr(value, "item"):
            value = value.item()
        log_items.append(f"{key}={float(value):.6g}")
    log_line = f"[Eval] step={global_step} | " + ", ".join(log_items)
    progress_bar.write(log_line)

    log_path = os.path.join(output_dir, "validation_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")


def unpack_image_depth(batch):
    if isinstance(batch, dict):
        return batch["image"], batch.get("depth")
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1]
    return batch, None


def main(args):
    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)

    train_dataset = H5Dataset(
        args.train_dataset,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.train_patch_size, args.train_patch_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]),
        depth_path=args.train_depth_dataset,
        paired_transform=ImageDepthTrainTransform(args.train_patch_size),
    )

    if args.test_depth_dir is not None:
        test_dataset = ImageDepthFolder(
            args.test_dataset,
            args.test_depth_dir,
            paired_transform=ImageDepthEvalTransform(args.train_patch_size),
        )
    else:
        test_dataset = ImageFolder(
            args.test_dataset,
            transform=transforms.Compose([
                transforms.Resize((args.train_patch_size, args.train_patch_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        pin_memory=True,
    )

    net = HMCGUIC(sd_path=sd_path, args=args)
    net.set_train()

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")
    if args.gradient_checkpointing:
        net.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_disc = vision_aided_loss.Discriminator(cv_type='dinov2_reg', output_type='conv_multi_level',
                                               loss_type=args.gan_loss_type, device="cuda")
    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)
    alex_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()
    alex_lpips.requires_grad_(False)
    # net_clip = CLIPLoss(clip_model_name='/your_local_dir/clip-vit-base-patch32').cuda()
    net_clip = CLIPLoss().cuda()
    net_structure = StructureLoss(
        lambda_sketch=args.lambda_structure_sketch,
        lambda_edge=args.lambda_structure_edge,
        lambda_lap=args.lambda_structure_lap,
    ).cuda()
    net_color = ColorLoss(
        lambda_chroma=args.lambda_color_chroma,
        lambda_mean=args.lambda_color_mean,
        lambda_vq=args.lambda_color_vq,
    ).cuda()
    net_semantic = SemanticLoss(
        lambda_attention=args.lambda_semantic_attention,
        lambda_boundary=args.lambda_semantic_boundary,
        lambda_vq=args.lambda_semantic_vq,
    ).cuda()
    net_machine_task = None
    if args.use_machine_task_loss:
        net_machine_task = MachineTaskLoss(
            cv_type=args.machine_task_cv_type,
            output_type=args.machine_task_output_type,
            loss_type=args.machine_task_loss_type,
            normalize_features=not args.disable_machine_task_feature_norm,
            layer_weights=args.machine_task_layer_weights,
            model_paths=args.machine_task_model_paths,
            model_classes=args.machine_task_model_classes,
            input_sizes=args.machine_task_input_sizes,
            checkpoint_keys=args.machine_task_checkpoint_keys,
            model_kwargs=args.machine_task_model_kwargs,
            normalize_types=args.machine_task_normalize_types,
            strict_load=args.machine_task_strict_load,
        ).cuda()

    layers_to_opt = list(net.codec.parameters())
    for n, _p in net.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net.unet.conv_in.parameters())
    for n, _p in net.vae.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=2e-5, betas=(args.adam_beta1, args.adam_beta2),
                                       weight_decay=args.adam_weight_decay, eps=args.adam_epsilon, )
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
                                      num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                      num_training_steps=args.max_train_steps * accelerator.num_processes)

    net, net_disc, optimizer, optimizer_disc, train_dataloader, lr_scheduler_disc = accelerator.prepare(net, net_disc,
                                                                                                        optimizer,
                                                                                                        optimizer_disc,
                                                                                                        train_dataloader,
                                                                                                        lr_scheduler_disc)
    net_lpips, alex_lpips, net_clip, net_structure, net_color, net_semantic = accelerator.prepare(
        net_lpips, alex_lpips, net_clip, net_structure, net_color, net_semantic
    )
    if net_machine_task is not None:
        net_machine_task = accelerator.prepare(net_machine_task)

    if accelerator.is_main_process:
        main_device = accelerator.device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    alex_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)
    net_structure.to(accelerator.device, dtype=weight_dtype)
    net_color.to(accelerator.device, dtype=weight_dtype)
    net_semantic.to(accelerator.device, dtype=weight_dtype)
    if net_machine_task is not None:
        net_machine_task.to(accelerator.device, dtype=weight_dtype)

    net_disc.to(accelerator.device, dtype=weight_dtype)
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
                        disable=not accelerator.is_local_main_process, )
    mse_loss = torch.nn.MSELoss()

    global_step = 0
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            batch, depth = unpack_image_depth(batch)
            if depth is not None:
                depth = depth.to(batch.device, dtype=batch.dtype)

            if global_step == 5000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 2e-5
            if global_step == 10000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5
            if global_step == 15000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-6
            l_acc = [net, net_disc]

            with accelerator.accumulate(*l_acc):
                B, C, H, W = batch.shape
                pos_tag_prompt = [1 for _ in range(B)]
                x_hat, RateLossOutput, aux_outputs = net(
                    batch,
                    pos_tag_prompt,
                    args.train_patch_size,
                    args.train_patch_size,
                    return_aux=True,
                )
                x = batch.detach().float()
                x_hat = x_hat.float()

                loss_l2 = mse_loss(x_hat, x)
                loss_lpips = net_lpips(x_hat, x).mean()
                loss_clip = net_clip(x_hat, x)
                loss_adv = net_disc(x_hat, for_G=True).mean()
                loss_D = loss_l2 * args.lambda_l2 + loss_lpips * args.lambda_lpips + loss_clip * args.lambda_clip + loss_adv * args.lambda_gan
                loss_structure, _ = net_structure(x_hat, x, aux_outputs["structure"])
                loss_color, _ = net_color(x_hat, x, aux_outputs["color"])
                loss_machine_task = x_hat.new_tensor(0.0)
                if net_machine_task is not None:
                    machine_info = aux_outputs["machine"]
                    if machine_info is None or machine_info.get("feature") is None:
                        raise RuntimeError("D_M did not produce F_hat_m.")
                    machine_input = machine_info["feature"].float()
                    loss_machine_task, _ = net_machine_task(machine_input, x, depth=depth)
                loss_semantic, _ = net_semantic(x_hat, x, aux_outputs["semantic"])
                loss_entropy_aux = accelerator.unwrap_model(net).codec.aux_loss()
                loss = (
                    RateLossOutput.rate_loss
                    + loss_D
                    + args.lambda_structure * loss_structure
                    + args.lambda_color * loss_color
                    + args.lambda_machine_task * loss_machine_task
                    + args.lambda_semantic * loss_semantic
                    + args.lambda_entropy_aux * loss_entropy_aux
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                loss_real = net_disc(x.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(loss_real)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

                loss_fake = net_disc(x_hat.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(loss_fake)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:

                    if (
                        global_step % args.checkpointing_steps == 0
                        or global_step == args.max_train_steps
                    ):
                        rate = f"{args.lambda_rate:g}"
                        outf = os.path.join(
                            args.output_dir,
                            "checkpoints",
                            f"hmcg_uic_lambda_{rate}_stage2_{global_step}.pkl",
                        )
                        accelerator.unwrap_model(net).save_model(outf)

                    if global_step % args.eval_freq == 1:
                        # Preserve the current training state during validation.
                        was_training = net.training

                        # Switch both networks to evaluation mode.
                        net.eval()
                        net_disc.eval()

                        l_rate, l_y, l_z, l_psnr, l_lpips = [], [], [], [], []
                        l_structure = []
                        l_color = []
                        l_machine_task = []
                        l_semantic = []
                        val_count = 0

                        with torch.no_grad():
                            for id, batch_val in enumerate(test_dataloader):
                                batch_val, depth_val = unpack_image_depth(batch_val)
                                batch_val = batch_val.to(main_device)
                                if depth_val is not None:
                                    depth_val = depth_val.to(main_device, dtype=batch_val.dtype)
                                B, C, H, W = batch_val.shape
                                assert B == 1, "Use batch size 1 for eval."

                                # Validation samples have already been resized by the dataset transform.
                                batch_val_input = batch_val

                                pos_tag_prompt = [1 for _ in range(B)]
                                x_hat, RateLossOutput, aux_outputs = accelerator.unwrap_model(net)(
                                    batch_val_input,
                                    pos_tag_prompt,
                                    args.train_patch_size,
                                    args.train_patch_size,
                                    return_aux=True,
                                )
                                loss_structure, _ = net_structure(
                                    x_hat.float(),
                                    batch_val_input.float(),
                                    aux_outputs["structure"],
                                )
                                l_structure.append(loss_structure.detach().item())
                                loss_color, _ = net_color(
                                    x_hat.float(),
                                    batch_val_input.float(),
                                    aux_outputs["color"],
                                )
                                l_color.append(loss_color.detach().item())
                                loss_semantic, _ = net_semantic(
                                    x_hat.float(),
                                    batch_val_input.float(),
                                    aux_outputs["semantic"],
                                )
                                l_semantic.append(loss_semantic.detach().item())

                                if net_machine_task is not None:
                                    machine_info = aux_outputs["machine"]
                                    if machine_info is None or machine_info.get("feature") is None:
                                        raise RuntimeError(
                                            "D_M did not produce F_hat_m during validation."
                                        )
                                    machine_input = machine_info["feature"].float()
                                    loss_machine_task, _ = net_machine_task(
                                        machine_input,
                                        batch_val_input.float(),
                                        depth=depth_val.float() if depth_val is not None else None,
                                    )
                                    l_machine_task.append(loss_machine_task.detach().item())

                                x_display = (batch_val_input * 0.5 + 0.5).float()
                                x_hat_display = (x_hat * 0.5 + 0.5).float()

                                loss_l2 = mse_loss(x_hat_display, x_display)
                                loss_psnr = 10 * (-torch.log(loss_l2) / np.log(10))
                                loss_lpips = alex_lpips(x_hat_display, x_display)

                                loss_R = RateLossOutput.quantized_total_bpp.detach()
                                loss_yrate = RateLossOutput.quantized_latent_bpp.detach()
                                loss_zrate = RateLossOutput.quantized_hyper_bpp.detach()

                                l_rate.append(loss_R.item())
                                l_y.append(loss_yrate.item())
                                l_z.append(loss_zrate.item())
                                l_lpips.append(loss_lpips.item())
                                l_psnr.append(loss_psnr.item())

                                if args.save_val and val_count < args.save_num:
                                    x_display = x_display.cpu().detach()
                                    x_hat_display = x_hat_display.cpu().detach()
                                    combined = torch.cat([x_display, x_hat_display], dim=3)
                                    output_pil = transforms.ToPILImage()(combined[0].clamp(0.0, 1.0))
                                    outf = os.path.join(args.output_dir, "eval", f"val_{id}.png")
                                    output_pil.save(outf)
                                    val_count += 1

                        logs = {}
                        logs["val/rate"] = np.mean(l_rate)
                        logs["val/y"] = np.mean(l_y)
                        logs["val/z"] = np.mean(l_z)
                        logs["val/psnr"] = np.mean(l_psnr)
                        logs["val/lpips"] = np.mean(l_lpips)
                        if len(l_structure) > 0:
                            logs["val/structure"] = np.mean(l_structure)
                        if len(l_color) > 0:
                            logs["val/color"] = np.mean(l_color)
                        if net_machine_task is not None and len(l_machine_task) > 0:
                            logs["val/machine_task"] = np.mean(l_machine_task)
                        if len(l_semantic) > 0:
                            logs["val/semantic"] = np.mean(l_semantic)
                        progress_bar.set_postfix(**logs)
                        write_eval_log(args.output_dir, global_step, logs, progress_bar)

                        # Restore the training state after validation.
                        if was_training:
                            net.train()
                            net_disc.train()

                        gc.collect()
                        torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)

            # Stop within the current dataloader epoch.  This makes Stage 2
            # exactly max_train_steps (21,001 in the paper configuration).
            if global_step >= args.max_train_steps:
                break


if __name__ == "__main__":
    args = parse_args_training()
    main(args)

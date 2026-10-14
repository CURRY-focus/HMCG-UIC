import argparse

def parse_args_testing(input_args=None):

    parser = argparse.ArgumentParser()

    # pretrained weights
    parser.add_argument("--sd_path", required=True, help="Path to SD-Turbo")
    parser.add_argument("--elic_path", required=True, help="Path to pretrained ELIC model")
    parser.add_argument("--codec_path", required=True, help="Path to pretrained HMCG-UIC weights", default=None)

    # testing images
    parser.add_argument("--img_path", type=str, required=True, default='/data/Kodak/')
    parser.add_argument("--max_images", type=int, default=None, help="Use only the first N sorted test images.")

    # output path
    parser.add_argument("--rec_path", type=str, required=True, default='/output/rec/')
    parser.add_argument("--bin_path", type=str, required=True, default='/output/bin/')
    parser.add_argument(
        "--machine_path",
        type=str,
        default=None,
        help="Optional directory for decoded machine-oriented tensors F_hat_m (saved as PNG task inputs).",
    )
    parser.add_argument(
        "--stream_bpp_csv",
        type=str,
        default=None,
        help="Optional CSV path for measured main/structure/color/semantic/container bpp.",
    )

    # model details
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=160)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--pos_prompt", type=str, default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")

    # testing details
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--lambda_rate", default=0.5, type=float)
    parser.add_argument("--structure_residual_scale", default=0.1, type=float)
    parser.add_argument("--color_codebook_size", default=256, type=int)
    parser.add_argument("--color_code_dim", default=64, type=int)
    parser.add_argument("--color_residual_scale", default=0.05, type=float)
    parser.add_argument(
        "--color_global_tokens",
        default=4,
        type=int,
        help="Use K global color tokens. The paper configuration uses 4 and must match training.",
    )
    parser.add_argument("--semantic_codebook_size", default=256, type=int)
    parser.add_argument("--semantic_code_dim", default=64, type=int)
    parser.add_argument("--semantic_residual_scale", default=0.05, type=float)
    parser.add_argument(
        "--semantic_global_tokens",
        default=4,
        type=int,
        help="Use K global semantic tokens. Must match training.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.require_paper_checkpoint = True

    return args

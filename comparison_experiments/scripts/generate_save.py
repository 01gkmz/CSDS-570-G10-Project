import argparse
import os
import numpy as np
import torch as th
from PIL import Image

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def save_images(images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)  # [-1, 1] → [0, 255]
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # [B, C, H, W] → [B, H, W, C]

    for i, img in enumerate(images):
        im = Image.fromarray(img)
        im.save(os.path.join(output_dir, f"sample_{i:03d}.png"))
        print(f"Saved: sample_{i:03d}.png")

def main():
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output_dir", type=str, default="./output", help="Where to save images")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of images to generate")
    args = parser.parse_args()

    dist_util.setup_dist()
    logger.configure()

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()

    batch_size = 4
    all_images = []
    while len(all_images) * batch_size < args.num_samples:
        model_kwargs = {}
        sample = diffusion.p_sample_loop(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )
        all_images.append(sample)

    all_images = th.cat(all_images, dim=0)[:args.num_samples]
    save_images(all_images, args.output_dir)

if __name__ == "__main__":
    main()

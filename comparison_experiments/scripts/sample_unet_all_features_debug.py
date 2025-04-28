import os
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import argparse

from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# 保存所有 hook 抓到的特征图
feature_maps = {}


def save_feature(name):
    def hook_fn(module, input, output):
        feature_maps[name] = output.detach().cpu()

    return hook_fn


def visualize_feature_map(feature_tensor, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)
    feature_np = feature_tensor[0].numpy()  # [C, H, W]
    for i, channel in enumerate(feature_np):
        plt.imshow(channel, cmap='viridis')
        plt.axis('off')
        plt.title(f'{prefix} - Channel {i}')
        plt.savefig(os.path.join(save_dir, f'{prefix}_ch{i}.png'))
        plt.close()


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        num_channels=128,
        num_res_blocks=3,
        learn_sigma=True,
        dropout=0.3,
        diffusion_steps=4000,
        noise_schedule='cosine',
        model_path="C:/Users/11427/PycharmProjects/improved-diffusion-main/checkpoints/cifar10_uncond_50M_500K.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    image_size = 32
    batch_size = 1
    steps_to_visualize = [1000, 2000, 3000, 4000]  # 🔵 多个 steps
    model_path = "C:/Users/11427/PycharmProjects/improved-diffusion-main/checkpoints/cifar10_uncond_50M_500K.pt"
    input_blocks_to_hook = list(range(16))
    output_blocks_to_hook = list(range(16))

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    print("Using device:", device)

    args = create_argparser().parse_args()

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    for idx in input_blocks_to_hook:
        name = f"input_block_{idx}"
        model.input_blocks[idx].register_forward_hook(save_feature(name))
    model.middle_block.register_forward_hook(save_feature("middle_block"))
    for idx in output_blocks_to_hook:
        name = f"output_block_{idx}"
        model.output_blocks[idx].register_forward_hook(save_feature(name))

    x = th.randn(batch_size, 3, image_size, image_size).to(device)

    os.makedirs("feature_vis", exist_ok=True)

    # 🔵 循环多个 steps
    for timestep in steps_to_visualize:
        print(f"\n--- Visualizing features at step {timestep} ---")

        t = th.tensor([timestep] * batch_size).to(device)

        feature_maps.clear()  # 每次清空上一次的 feature map
        with th.no_grad():
            _ = model(x, t)

        step_dir = f"feature_vis/step_{timestep}"
        os.makedirs(step_dir, exist_ok=True)

        for name, feat in feature_maps.items():
            print(f"{name} shape: {feat.shape}")
            np.save(os.path.join(step_dir, f"{name}.npy"), feat[0].numpy())
            visualize_feature_map(feat, save_dir=os.path.join(step_dir, name), prefix=name)

    print("\nAll features saved to 'feature_vis/'")


if __name__ == "__main__":
    main()

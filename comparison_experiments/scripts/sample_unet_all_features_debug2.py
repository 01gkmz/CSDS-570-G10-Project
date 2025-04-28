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


def save_image(img_tensor, save_path):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    img = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)  # [-1,1] -> [0,255]
    plt.imsave(save_path, img)


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
    steps_to_visualize = [4000]  # 只保存4000步的特征图
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

    # 🔵 生成扩散过程最终图片
    print("\n--- Generating final denoised image ---")
    sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    model_input = th.randn(batch_size, 3, image_size, image_size).to(device)

    with th.no_grad():
        final_samples = sample_fn(
            model,
            (batch_size, 3, image_size, image_size),
            clip_denoised=args.clip_denoised
        )

    save_image(final_samples[0], "feature_vis/final_generated_image.png")
    print("Final generated image saved to 'feature_vis/final_generated_image.png'")

    # 🔵 绘制特征图 (这里仍然使用4000步)
    for timestep in steps_to_visualize:
        print(f"\n--- Visualizing features at step {timestep} ---")

        t = th.tensor([timestep] * batch_size).to(device)

        feature_maps.clear()
        with th.no_grad():
            _ = model(x, t)

        step_dir = f"feature_vis/step_{timestep}"
        os.makedirs(step_dir, exist_ok=True)

        for name, feat in feature_maps.items():
            print(f"{name} shape: {feat.shape}")
            np.save(os.path.join(step_dir, f"{name}.npy"), feat[0].numpy())
            visualize_feature_map(feat, save_dir=os.path.join(step_dir, name), prefix=name)

    print("\nAll features and final generated image saved to 'feature_vis/'")


if __name__ == "__main__":
    main()

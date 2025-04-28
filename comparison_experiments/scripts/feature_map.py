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

feature_maps = {}


def save_feature(name):
    def hook_fn(module, input, output):
        feature_maps[name] = output.detach().cpu()

    return hook_fn


def visualize_feature_map(feature_tensor, save_dir, prefix="middle_block"):
    os.makedirs(save_dir, exist_ok=True)
    feature_np = feature_tensor[0].numpy()
    # feature_np = feature_tensor[0, :8].numpy()  # 前8个通道
    for i, channel in enumerate(feature_np):
        plt.imshow(channel, cmap='viridis')
        plt.axis('off')
        plt.title(f'{prefix} - Channel {i}')
        plt.savefig(os.path.join(save_dir, f'{prefix}_ch{i}.png'))
        plt.close()


def visualize_feature_map2(feature_tensor, save_dir, prefix="input_blocks"):
    os.makedirs(save_dir, exist_ok=True)
    feature_np = feature_tensor[0].numpy()
    # feature_np = feature_tensor[0, :8].numpy()  # 前8个通道
    for i, channel in enumerate(feature_np):
        plt.imshow(channel, cmap='viridis')
        plt.axis('off')
        plt.title(f'{prefix} - Channel {i}')
        plt.savefig(os.path.join(save_dir, f'{prefix}_ch{i}.png'))
        plt.close()


def visualize_feature_map3(feature_tensor, save_dir, prefix="output_blocks"):
    os.makedirs(save_dir, exist_ok=True)
    feature_np = feature_tensor[0].numpy()
    # feature_np = feature_tensor[0, :8].numpy()  # 前8个通道
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
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    # 基本配置
    image_size = 32
    batch_size = 1
    timestep = 500  # 想观察的扩散步骤

    # 加载模型
    model_path = "C:/Users/11427/PycharmProjects/improved-diffusion-main/checkpoints/cifar10_uncond_50M_500K.pt"

    args = create_argparser().parse_args()

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    print(dist_util.dev())

    # 注册 hook，抓取 middle_block 的特征图
    model.middle_block.register_forward_hook(save_feature("middle_block"))
    model.input_blocks.register_forward_hook(save_feature("input_blocks"))

    # 准备输入
    x = th.randn(batch_size, 3, image_size, image_size).to(dist_util.dev())
    t = th.tensor([timestep] * batch_size).to(dist_util.dev())

    # 一次推理
    with th.no_grad():
        _ = model(x, t)

    # 获取特征图
    feat = feature_maps["middle_block"]
    print("Feature map shape:", feat.shape)  # e.g. [B, C, H, W]

    feat_input = feature_maps["input_blocks"]
    print("Feature map shape:", feat_input.shape)

    feat_output = feature_maps["output_blocks"]
    print("Feature map shape:", feat_output.shape)

    # 保存特征图为 numpy
    np.save("middle_block_features.npy", feat[0].numpy())
    np.save("input_blocks_features.npy", feat_input[0].numpy())
    np.save("output_blocks_features.npy", feat_input[0].numpy())

    # 可视化并保存为图像
    visualize_feature_map(feat, save_dir="feature_vis/middle", prefix="middle_block")

    visualize_feature_map2(feature_maps, save_dir="feature_vis/input", prefix="input_blocks")

    visualize_feature_map3(feature_maps, save_dir="feature_vis/output", prefix="input_blocks")

    print("Saved feature maps to 'feature_vis/' and middle_block_features.npy")


if __name__ == "__main__":
    main()

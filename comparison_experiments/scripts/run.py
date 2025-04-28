import torch
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils
import os
from PIL import Image

# 参数设置
image_size = 32
num_steps = 10  # 展示时间步数量
save_dir = "./diffusion_noise_steps"
os.makedirs(save_dir, exist_ok=True)

# β 线性调度
def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# 计算 diffusion 参数
def get_diffusion_params(timesteps):
    betas = linear_beta_schedule(timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

# 前向加噪函数 q(x_t | x_0)
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    noise = torch.randn_like(x_start)
    x_t = (
        sqrt_alphas_cumprod[t][:, None, None, None] * x_start +
        sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise
    )
    return x_t, noise

# 取一张图像
def get_sample_image():
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),  # [0, 1]
        T.Normalize((0.5,), (0.5,)),  # [-1, 1]
    ])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    img, _ = dataset[0]
    return img.unsqueeze(0)  # shape: [1, 3, H, W]

# 保存图像
def save_tensor_as_image(tensor, path):
    tensor = tensor.clone()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)  # unnormalize [-1,1] -> [0,1]
    vutils.save_image(tensor, path)

# 主逻辑
def main():
    timesteps = 1000
    betas, alphas, alphas_cumprod = get_diffusion_params(timesteps)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    x_start = get_sample_image()  # [1, 3, 32, 32]
    x_start = x_start.to(torch.float32)

    step_indices = torch.linspace(0, timesteps - 1, num_steps).long()

    for idx, t in enumerate(step_indices):
        t_batch = torch.tensor([t])
        x_t, noise = q_sample(x_start, t_batch, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

        # 保存加噪图像
        noisy_path = os.path.join(save_dir, f"noisy_t{t.item():04d}.png")
        save_tensor_as_image(x_t[0], noisy_path)

        # 保存噪声图像
        noise_path = os.path.join(save_dir, f"noise_t{t.item():04d}.png")
        save_tensor_as_image(noise[0], noise_path)

        print(f"[t={t.item():04d}] Saved noisy image and noise to:")
        print(f" - {noisy_path}")
        print(f" - {noise_path}")

    print("✅ 所有步骤完成。图像与噪声已保存。")

if __name__ == "__main__":
    main()

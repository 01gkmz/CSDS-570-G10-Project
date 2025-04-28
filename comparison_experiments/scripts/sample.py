import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# -------- 模型定义 --------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.enc1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.enc1(x))  # enc1
        x = self.pool2(self.enc2(x))  # enc2
        x = self.pool3(self.enc3(x))  # enc3
        x = self.bottleneck(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -------- 加载模型和数据 --------
device = torch.device("cpu")
model = UNetClassifier().to(device)
model.load_state_dict(torch.load('C:/Users/11427/PycharmProjects/improved-diffusion-main/checkpoints/best.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# -------- Hook 函数 --------
feature_maps = {}


def save_feature_map(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach().cpu()

    return hook


# 注册 hook
model.enc1.register_forward_hook(save_feature_map('enc1'))
model.enc2.register_forward_hook(save_feature_map('enc2'))
model.enc3.register_forward_hook(save_feature_map('enc3'))
model.bottleneck.register_forward_hook(save_feature_map('bottleneck'))

# -------- 推理一张图像以提取特征图 --------
image, label = testset[0]
image = image.unsqueeze(0).to(device)
_ = model(image)


# -------- 保存所有通道的特征图为单独图像 --------
def save_all_feature_maps(tensor, layer_name):
    output_dir = f"feature_vis2/{layer_name}"
    os.makedirs(output_dir, exist_ok=True)

    num_channels = tensor.shape[1]
    for i in range(num_channels):
        channel = tensor[0, i]
        plt.imshow(channel, cmap='viridis')
        plt.axis('off')
        plt.savefig(f"{output_dir}/channel_{i:03d}.png", bbox_inches='tight', pad_inches=0)
        plt.close()


# 保存每一层的所有通道特征图
os.makedirs("feature_vis2", exist_ok=True)
for name, fmap in feature_maps.items():
    print(f"Saving features from {name} ({fmap.shape[1]} channels)")
    save_all_feature_maps(fmap, name)

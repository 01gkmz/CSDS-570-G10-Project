from improved_diffusion.unet import UNetModel
from torchviz import make_dot
import torch

# 模拟输入
x = torch.randn(1, 3, 32, 32)
t = torch.tensor([0])  # 时间步输入
model = UNetModel(
    # image_size=32,
    in_channels=3,
    model_channels=128,
    out_channels=3,
    num_res_blocks=2,
    attention_resolutions=[16],
    dropout=0.1,
    channel_mult=(1, 2, 2),
    num_classes=None,
    use_checkpoint=False,
    # use_fp16=False,
    num_heads=4,
    # num_head_channels=64,
    use_scale_shift_norm=True,
)

# 前向传播
model.eval()
x_out = model(x, t)

# 可视化
dot = make_dot(x_out, params=dict(list(model.named_parameters())))
dot.render("unet_structure", format="png")  # 会生成 unet_structure.png

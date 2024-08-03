import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import Generator  # 从 models.py 导入 Generator

# 超参数
latent_dim = 100
num_images = 16  # 生成图像的数量

# 初始化生成器
generator = Generator()

# 加载训练后的权重
state_dict = torch.load('generator.pth', map_location=torch.device('cpu'))
generator.load_state_dict(state_dict)
generator.eval()  # 设置为评估模式

# 创建 image 文件夹
if not os.path.exists('image'):
    os.makedirs('image')

# 生成随机噪声向量
z = torch.randn(num_images, latent_dim)

# 生成图像
with torch.no_grad():  # 禁用梯度计算
    generated_imgs = generator(z)

# 将生成的图像转换为 [0, 1] 范围
generated_imgs = (generated_imgs + 1) / 2.0  # 将 [-1, 1] 范围的图像值转为 [0, 1]

# 将图像转换为 PIL 格式以便保存和显示
to_pil = transforms.ToPILImage()

# 显示生成的图像
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    img_pil = to_pil(generated_imgs[i].cpu().squeeze(0))
    ax.imshow(img_pil, cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()

# 保存生成的图像到 image 文件夹中
for i in range(num_images):
    img_pil = to_pil(generated_imgs[i].cpu().squeeze(0))
    img_pil.save(f'image/generated_image_{i}.png')

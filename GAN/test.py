import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 超参数
batch_size = 64
lr = 0.0002
epochs = 10
latent_dim = 100

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 获取数据集的一个批次
data_iter = iter(train_loader)
images, labels = next(data_iter)  # 获取一个批次的数据


# 将图像数据从 [-1, 1] 转换为 [0, 1]
def denorm(imgs):
    return imgs / 2 + 0.5


# 显示前几张图片
def show_images(images, labels, num_images=8):
    images = denorm(images)  # 反归一化
    images = images.numpy()  # 转换为 NumPy 数组
    fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
    for i in range(num_images):
        axes[i].imshow(np.squeeze(images[i]), cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')
    plt.show()


# 显示前 8 张图片
show_images(images, labels, num_images=8)

import argparse
import os
import torch
from model import Generator, Discriminator,GANOptions
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from torchvision import datasets
from torch.autograd import Variable
# 获取解析后的参数
opt = GANOptions().parse()
os.makedirs("images", exist_ok=True)
print(opt)
img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)
# 创建 DataLoader 实例
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",  # 数据集存储路径
        train=True,  # 是否加载训练集
        download=True,  # 如果数据集不存在，则下载
        transform=transforms.Compose(  # 数据预处理
            [
                transforms.Resize(opt.img_size),  # 调整图像大小
                transforms.ToTensor(),  # 转换为 Tensor
                transforms.Normalize([0.5], [0.5])  # 归一化
            ]
        ),
    ),
    batch_size=opt.batch_size,  # 每个批次的样本数量
    shuffle=True,  # 是否打乱数据
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# b1，一阶矩估计指的是对梯度的均值进行估计。它类似于动量优化中的动量项，用于平滑梯度的变化，并减少震荡
# b2，二阶矩估计指的是对梯度的方差（或平方）进行估计。它用于调整每个参数的学习率，以避免学习率过高或过低

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    #dataloader 提供了批量的训练数据，每批数据包含图片 imgs 和标签 _（标签在这里未被使用）
    for i, (imgs, _) in enumerate(dataloader):
        # Adversarial ground truths，定义真实和虚假的标签
        # Variable 用于包装张量，使其可以用于计算图梯度。requires_grad=False 表示这些张量不会计算梯度，老版本的torch代码
        # 下面有新版本的代码
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        # 这行代码的目的是创建一个形状为 [batch_size, 1] 的张量，并用值 1.0 填充整个张量。张量的所有元素都将是 1.0。
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
# 保存模型
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

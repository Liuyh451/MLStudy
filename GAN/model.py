import argparse
import numpy as np
import torch.nn as nn


class GANOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
        self.parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
        self.parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        self.parser.add_argument("--b1", type=float, default=0.5,
                                 help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--b2", type=float, default=0.999,
                                 help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--n_cpu", type=int, default=8,
                                 help="number of cpu threads to use during batch generation")
        self.parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
        self.parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
        self.parser.add_argument("--channels", type=int, default=1, help="number of image channels")
        self.parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")

    def parse(self):
        return self.parser.parse_args()
# 获取解析后的参数
opt = GANOptions().parse()
img_shape = (opt.channels, opt.img_size, opt.img_size)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 定义了一个神经网络层的构建模块，用于创建由线性层、批量归一化层和激活函数组成的序列
        # in_feat表示输入特征的维度，即输入到线性层的特征数
        # out_feat输出特征的维度，即线性层输出的特征数
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            # 是否在线性层后添加批量归一化层，批量归一化层有助于加快网络训练和稳定性。这里的 0.8 是动量参数（momentum），用于控制移动平均的更新速率
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            # 添加激活层， 0.2 是负斜率参数，inplace=True 表示是否在原地执行操作，从而节省内存
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            # 返回网络结构
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            # np.prod(img_shape)展平图像并传递给全连接层，可以有效地提取和处理图像的全局特征
            # 并不是所有网络都要这个操作，LSTM就不需要，根据情况判断
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    # 前向传播
    def forward(self, z):
        img = self.model(z)
        # 将 img 张量的形状进行调整，img_shape上面有定义
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

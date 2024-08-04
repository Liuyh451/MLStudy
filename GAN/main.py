import numpy as np
import torch
from model import GANOptions
from torchvision.utils import save_image
from torch.autograd import Variable

from model import Generator  # 从 models.py 导入 Generator
# 获取解析后的参数
opt = GANOptions().parse()
img_shape = (opt.channels, opt.img_size, opt.img_size)
# 初始化生成器并加载训练好的权重
generator = Generator()
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

# 检查CUDA是否可用，并将生成器移动到CUDA设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
# 生成随机噪声向量
z = torch.randn(opt.batch_size, opt.latent_dim).to(device)
#用下面的方式生成噪声向量z需要加上tensor=那行代码
# cuda = True if torch.cuda.is_available() else False
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
# 生成假图像
with torch.no_grad():  # 不需要计算梯度
    gen_imgs = generator(z)
save_image(gen_imgs.data[:25], "images/A.png" , nrow=5, normalize=True)
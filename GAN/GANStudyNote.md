# GANStudyNote

## 1.数据集

来自torch自带的datasets.MNIST数据集

训练集：90000张图片28*28

测试集：10000张图片28*28

## 2.数据预处理

**`datasets.MNIST`**:

- `datasets.MNIST` 是 PyTorch 提供的一个内置数据集类，用于加载 MNIST 数据集。
- `./data/mnist` 是数据集的存储路径，如果路径下不存在数据集，`download=True` 会自动下载数据集到该路径。
- `train=True` 表示加载训练集。将其设置为 `False` 可以加载测试集。

**`transform=transforms.Compose(...)`**:

- `transforms.Compose` 用于将多个图像变换操作组合在一起。
- `transforms.Resize(opt.img_size)`：调整图像的大小，`opt.img_size` 应该是一个整数，表示目标尺寸（如 `(64, 64)`）。
- `transforms.ToTensor()`：将图像转换为 PyTorch 的 `Tensor` 类型，像素值会被缩放到 `[0, 1]` 范围内。
- `transforms.Normalize([0.5], [0.5])`：对图像进行归一化处理。图像像素值被减去均值 `0.5`，然后除以标准差 `0.5`。这是为了将像素值调整到 `[-1, 1]` 范围内，有助于训练过程中的收敛。

**`torch.utils.data.DataLoader`**:

- `DataLoader` 是 PyTorch 提供的用于批处理和数据迭代的工具。
- `batch_size=opt.batch_size`：指定每个批次的样本数量。`opt.batch_size` 是一个变量，应该在代码的其他地方定义（例如，通过命令行参数）。
- `shuffle=True`：表示在每个训练周期开始时，数据会被随机打乱。这有助于提高模型的泛化能力。

## 3. 网络结构

### 3.1 block

1. **线性层**：`nn.Linear(in_feat, out_feat)` 创建一个全连接（线性）层，将输入的特征数从 `in_feat` 映射到 `out_feat`。
2. **批量归一化层**（如果 `normalize` 为 `True`）：`nn.BatchNorm1d(out_feat, 0.8)` 添加一个批量归一化层，对输入到该层的数据进行归一化，即减去均值并除以标准差，然后应用一个缩放和平移操作，批量归一化层有助于加快网络训练和稳定性。这里的 `0.8` 是动量参数（momentum），用于控制移动平均的更新速率。
3. **激活函数**：`nn.LeakyReLU(0.2, inplace=True)` 添加一个 LeakyReLU 激活函数。LeakyReLU 是一种带有小负斜率的修正线性单元（ReLU），用于处理输入小于零的情况。这里的 `0.2` 是负斜率参数，`inplace=True` 表示是否在原地执行操作，从而节省内存。
4. **返回层的列表**：将构建好的层列表返回。

### 3.2 张量重塑

```python
img = img.view(img.size(0), *img_shape)
```

假设 `img_shape` 是一个包含图像维度的元组，例如 `(channels, height, width)`。那么，这行代码会将 `img` 张量重塑为形状 `(batch_size, channels, height, width)`，其中 `batch_size` 是输入批次的大小

### 3.3 张量展平

`np.prod(img_shape)` 是 NumPy 中的一个函数，用于计算给定形状数组中所有元素的乘积。在这个上下文中，它通常用于计算展平张量所需的总元素数量。

```python
img_shape = (3, 64, 64)
# 计算 img_shape 中所有元素的乘积
num_elements = np.prod(img_shape)
print(num_elements)  # 输出 12288
```

在神经网络模型中，尤其是生成对抗网络（GAN）或自动编码器中，可能需要将一个扁平化的向量（如潜在空间向量）重塑为具有特定形状的图像。我们将图像张量展平成一维向量，使其适合于全连接层的输入。展平图像并传递给全连接层，可以有效地提取和处理图像的全局特征。（并不是所有网络都需要例如LSTM）

### 3.4 优化器

**`torch.optim.Adam`**:

- `Adam` 是一种自适应学习率优化算法，结合了动量和自适应梯度算法，能够在训练中动态调整每个参数的学习率。它常用于深度学习中的各种任务，特别是在处理大量数据和高维参数时。

**`discriminator.parameters()`**:

- `discriminator.parameters()` 返回判别器模型的所有参数。这些参数将由优化器进行更新，以最小化损失函数。
- `discriminator` 是定义好的判别器模型实例，它继承自 PyTorch 的 `nn.Module` 类。

**`lr=opt.lr`**:

- `lr`（learning rate，学习率）是控制模型参数更新步长的超参数。`opt.lr` 是从配置中获取的学习率值。较大的学习率可能导致训练不稳定，而较小的学习率则可能导致收敛速度慢。

**`betas=(opt.b1, opt.b2)`**:

- `betas` 是 Adam 优化器的两个动量项的系数，分别用于计算一阶矩估计（均值）和二阶矩估计（方差）。
-  `opt.b1`和`opt.b2`是从配置中获取的动量系数：
  - `beta1`（`opt.b1`）通常设为接近于 1 的值，如 0.9。
  - `beta2`（`opt.b2`）通常设为接近于 1 的值，如 0.999。
- 这些值影响梯度的平滑程度，`beta1` 控制一阶矩（均值）的衰减速度，`beta2` 控制二阶矩（方差）的衰减速度。

公式略

## 4. 训练

`dataloader` 提供了批量的训练数据，每批数据包含图片 `imgs` 和标签 `_`（标签在这里未被使用）

```
valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
```

- `valid` 是真实图像的标签，设为 1.0。
- `fake` 是生成图像的标签，设为 0.0。
- `Variable` 用于包装张量，使其可以用于计算图梯度。`requires_grad=False` 表示这些张量不会计算梯度。
- 这行代码的目的是创建一个形状为 `[batch_size, 1]` 的张量，并用值 `1.0` 填充整个张量。张量的所有元素都将是 `1.0`。
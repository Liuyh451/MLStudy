# LSTMStudyNotes

## 1.单特征时间序列lstm

### 1.1 数据集

股价预测[阿里天池茅台股价数据](https://tianchi.aliyun.com/dataset/165327)，仅取时间列和闭盘列

### 1.2 模型

#### 1.2.1 bili菜老师的模型

遇到的问题

模型loss值较大，且预测值一直徘徊在某个特定值，且不到真实值的一半

#### 1.2.2 在上模型上根据gpt更改的模型

在调优无果后，观察了数据集，发现数据变化幅度较大，从起初的80到最后1200，期间仅经历3k数据。于是怀疑数据没有预处理好。

##### 1.StandardScaler 标准化

它会去除数据的均值，并缩放到单位方差（即标准差为 1）`fit_transform` 是 `StandardScaler` 的一个方法，结合了 `fit` 和 `transform` 两个步骤。

`fit` 方法计算数据的均值和标准差，并保存这些统计量以便后续使用，`transform` 方法根据保存的均值和标准差，将数据进行标准化。

所以**`scaler.fit_transform(df[['close']])`**:这一步计算 `close` 列的均值和标准差，并将其转换为标准化数据

```python
data = {
    'close': [1.0, 2.0, 3.0, 4.0, 5.0]
}
scaler = StandardScaler()
df['close'] = scaler.fit_transform(df[['close']])#略去了一些东西
```

比如上面的代码执行结果为

```arduino
      close
0 -1.414214
1 -0.707107
2  0.000000
3  0.707107
4  1.414214
```

标准化的目的是将数据调整到一个标准范围内，减少特征之间的量级差异，通常有助于机器学习算法更好地收敛和提高模型性能

**逆标准化**：通过上述步骤得到的预测结果是正态类型（如上面的输出结果所示）的，需要转为源数据

```python
test_outputs_original = scaler.inverse_transform(test_outputs)
```

##### 2. 数据集划分

通常将数据集划分为 `train`，`val`，`test`，即测试集，验证集和测试集

并且把每个数据集如train，划分为x，y，如下

```python
X_train = torch.FloatTensor(train[['close']].values).view(-1, 1, 1)
y_train = torch.FloatTensor(train[['close']].values)
```

实际上x和y数据一样且都是tensor，但是x为3D维度，y为2D维度，gpt说了一大堆，没看太明白。但是我的理解是3D数据可以直接丢给模型训练并且预测，当需要用到源数据时，一般用y（2D）

##### 3. 超参数调优

 `hidden_size`，`num_layers`，`num_epochs`，通过多次调优发现：

1.hidden_size固定，num_epochs固定，增加隐藏层num_layers的数量，预测结果反而**下降**

2..hidden_size固定，num_layers固定，增加训练次数num_epochs，预测结果**小幅度**上升

3.num_layers固定，训练次数num_epochs固定，增加细胞数量，预测结果**明显**上升

所以隐藏层num_layers固定为1（增加就会降），步长100调整细胞数量，500增加epo，不断调优

## 3. LSTM基础知识

### 3.1 样本格式（input和out）

```
[batch_size,sequence_length,feature_size]
```

假设处理一个时间序列数据集，其中每个样本包含 30 天的股票价格数据

`batch_size` 是每批数据的样本数量，为n就是处理一次处理n个样本，

**内存：**较小的 `batch_size` 可以减少每次训练时所需的内存，但可能会导致训练时间增加。较大的 `batch_size` 可以加快训练过程，但需要更多的内存。

**训练稳定性**: 较小的 `batch_size` 可以导致训练过程中更多的梯度波动，而较大的 `batch_size` 可以提供更平稳的梯度估计，但也可能导致训练的泛化能力下降

`sequence_length` 是每个序列的长度，这里是 30。

`feature_size` 是每个时间步的特征数量（例如，价格、成交量等）






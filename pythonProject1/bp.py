import numpy as np
from sklearn.preprocessing import LabelBinarizer

# logistic回归
def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))

# 构建神经网络
class NeuralNetwork:
    def predict(self, x):
        for b, w in zip(self.biases, self.weights):
            # 计算权重相加再加上偏向的结果
            z = np.dot(x, w) + b
            # 计算输出值
            x = self.activation(z)
        return self.classes_[np.argmax(x, axis=1)]


# 构建BP神经网络
class BP(NeuralNetwork):
    def __init__(self, layers, batch):
        self.layers = layers
        self.batch = batch
        self.activation = logistic
        self.activation_deriv = logistic_derivative
        self.num_layers = len(layers)
        self.biases = [np.random.randn(x) for x in layers[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(layers[:-1], layers[1:])]

    def fit(self, X, y, learning_rate=0.1, epochs=1):
        labelbin = LabelBinarizer()
        y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        training_data = [(x, y) for x, y in zip(X, y)]
        n = len(training_data)
        for k in range(epochs):
            # 每次迭代都循环一次训练
            batches = [training_data[k:k + self.batch] for k in range(0, n, self.batch)]
            # 批量梯度下降
            for mini_batch in batches:
                x = []
                y = []
                for a, b in mini_batch:
                    x.append(a)
                    y.append(b)
                activations = [np.array(x)]
                # 向前一层一层的走
                for b, w in zip(self.biases, self.weights):
                    # 计算激活函数的参数,计算公式：权重.dot(输入)+偏向
                    z = np.dot(activations[-1], w) + b
                    # 计算输出值
                    output = self.activation(z)
                    # 将本次输出放进输入列表，后面更新权重的时候备用
                    activations.append(output)
                # 计算误差值
                error = activations[-1] - np.array(y)
                # 计算输出层误差率
                deltas = [error * self.activation_deriv(activations[-1])]
                # 循环计算隐藏层的误差率,从倒数第2层开始
                for l in range(self.num_layers - 2, 0, -1):
                    deltas.append(self.activation_deriv(activations[l]) * np.dot(deltas[-1], self.weights[l].T))

                # 将各层误差率顺序颠倒，准备逐层更新权重和偏向
                deltas.reverse()
                # 更新权重和偏向
                for j in range(self.num_layers - 1):
                    # 权重的增长量，计算公式，增长量 = 学习率 * (错误率.dot(输出值)),单个训练数据的误差
                    delta = learning_rate / self.batch * (
                        (np.atleast_2d(activations[j].sum(axis=0)).T).dot(np.atleast_2d(deltas[j].sum(axis=0))))
                    # 更新权重
                    self.weights[j] -= delta
                    # 偏向增加量，计算公式：学习率 * 错误率
                    delta = learning_rate / self.batch * deltas[j].sum(axis=0)
                    # 更新偏向
                    self.biases[j] -= delta
        return self


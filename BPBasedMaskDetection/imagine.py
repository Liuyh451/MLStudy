import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

#1.切分训练集和测试集
X = []  # 定义图像名称
Y = []  # 定义图像分类类标
Z = []  # 定义图像像素
# 记得更改此处4或者10
for i in range(0, 4):
    for f in os.listdir("photo" % i):
        # 获取图像名称
        X.append("photo//" + str(i) + "//" + str(f))
        # 获取图像类标即为文件夹名称
        Y.append(i)

X = np.array(X)
Y = np.array(Y)
print(X)
print(Y)
# 随机率为100% 选取其中的20%作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.2, random_state=1)

print(len(X_train), len(X_test), len(y_train), len(y_test))
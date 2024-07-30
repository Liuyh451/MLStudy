import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
import cv2
from bp import BP
import torch



starttime = datetime.datetime.now()

X = []
Y = []
X_local = []

for i in range(0, 2):
    # 遍历文件夹，读取图片
    for f in os.listdir("D:/python/lyh/MachineLearning/pythonProject1/photo/%s" % i):
        X.append("D:/python/lyh/MachineLearning/pythonProject1/photo//" + str(i) + "//" + str(f))
        Y.append(i)
X = np.array(X)
Y = np.array(Y)
# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

# 后续显示图片
X_local = X_test
X_train1 = X_train
X_test1 = X_test
X_test = []
X_train = []


# 进行图片并灰度化
def gray(X_train1, X_test1, X_train, X_test):
    for i in X_train1:
        Images = cv2.imread(i)
        image = cv2.resize(Images, (256, 256), interpolation=cv2.INTER_CUBIC)
        hist = cv2.calcHist([image], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
        X_train.append(((hist / 255).flatten()))

    for i in X_test1:
        Images = cv2.imread(i)
        image = cv2.resize(Images, (256, 256), interpolation=cv2.INTER_CUBIC)
        hist = cv2.calcHist([image], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
        X_test.append(((hist / 255).flatten()))


gray(X_train1, X_test1, X_train, X_test)
X_train = np.array(X_train)
X_test = np.array(X_test)
clf0 = BP([X_train.shape[1], 2], 2).fit(X_train, y_train, epochs=100)
predictions_labels = clf0.predict(X_test)
print("混淆矩阵为:")
print(confusion_matrix(y_test, predictions_labels))
result = classification_report(y_test, predictions_labels)
print("结果为:")
print(result)
endtime = datetime.datetime.now()
print(endtime - starttime)


# 输出前10张图片及预测结果
def weapon_name(num):
    numbers = {
        0: "不戴口罩",
        1: "戴口罩"
    }
    return numbers.get(num, None)


def picpredictio_example(X0_test):
    # 读取图像
    print(X0_test[k])
    image = cv2.imread(X0_test[k])
    # 显示图像
    print(weapon_name(predictions_labels[k]))
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 1


print("展示测试集前10张图片的预测结果")
k = 0
while k < 10:
    picpredictio_example(X_test1)
    k = k + 1

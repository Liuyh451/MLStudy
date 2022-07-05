from sklearn.preprocessing import binarize
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
import cv2
import os
from sklearn. model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import datetime
starttime = datetime.datetime.now()


X = []  # 用于存放图片的图像矩阵
Y = []  # 用于存放标签矩阵


def get_image(X, Y):  # 获取数据集中的图片并保存在数组中
    cnt = 0
    for i in range(0, 4):
        # 遍历文件夹
        for f in os.listdir("./photo/%s" % i):  # 打开文件夹，读取图片
            cnt += 1  # 统计数据集图片数量
            X.append("photo//" + str(i) + "//" + str(f))  # 图片放入列表
            Y.append(i)
    # 将列表转为数组，便于操作
    X = np.array(X)
    Y = np.array(Y)
    print("该数据集中一共{name}张图片".format(name=cnt))
    return X, Y
get_image(X, Y)


def dataset_split(X, Y):
    global X0_train, X0_test, y0_train, y0_test
    # 将数据集拆分为训练集和测试集，测试集比例为0.3，采用随机拆分，每一次测试集和训练集均不同
    X0_train, X0_test, y0_train, y0_test = train_test_split(
        X, Y, test_size=0.3, random_state=1)
    print("训练集一共{s}张图片,训练集一共{t}张图片".format(s=len(X0_train), t=len(X0_test)))
    return X0_train, X0_test, y0_train, y0_test


dataset_split(X, Y)

X_train = []  # 创建空列表存放数据

# 对要训练的图片进行数字图像处理，提取特征


def trainpic_process(X0_train):
    for i in X0_train:
        # 读取图像
        image = cv2.imread(i)

        # 图像像素大小一致
        img = cv2.resize(image, (256, 256),
                         interpolation=cv2.INTER_CUBIC)
        # 将图片格式从jpg转为hsv，便于生成2D直方图
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 计算图像直方图并存储至X数组，图片特征为H 色调值（0，180），S饱和度（0，256）
        hist = cv2.calcHist([hsv], [0, 1], None,
                            [180, 256], [0.0, 180.0, 0.0, 256.0])

        X_train.append(hist.flatten())
    return X_train


X_train = trainpic_process(X0_train)

X_test = []  # 创建空列表存放数据


def testpic_process(X0_test):
    # 测试集
    for i in X0_test:
        # 读取图像
        # print i
        image = cv2.imread(i)

        # 图像像素大小一致
        img = cv2.resize(image, (256, 256),
                         interpolation=cv2.INTER_CUBIC)
        # 将图片格式从jpg转为hsv，便于生成2D直方图
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 计算图像直方图并存储至X数组，图片特征为H 色调值（0，180），S饱和度（0，256）
        hist = cv2.calcHist([hsv], [0, 1], None,
                            [180, 256], [0.0, 180.0, 0.0, 256.0])

        X_test.append(hist.flatten())
    return X_test


X_test = testpic_process(X0_test)


class MLE_Predict:
    def predict(self, x):  # 创建求最大释然估计的类方法
        # 预测标签
        X = binarize(x, threshold=self.threshold)
        # print(X)
        # 使对数似然函数最大的值也使似然函数最大
        #Y_predict = np.dot(X, np.log(prob).T)+np.dot(np.ones((1,prob.shape[1]))-X, np.log(1-prob).T)
        # 等价于  lnf(x)=xlnp+(1-x)ln(1-p)
        Y_predict = np.dot(X, np.log(self.prob).T)-np.dot(X,
                                                          np.log(1-self.prob).T) + np.log(1-self.prob).sum(axis=1)
        return self.classes[np.argmax(Y_predict, axis=1)]
        # argmax(f(x))是使得f(x)取得最大值所对应的变量点x(或x的集合)


class Naive_Bayes(MLE_Predict):
    # 创建实例变量，确定分割阈值和先验概率
    def __init__(self, threshold):
        self.threshold = threshold
        self.classes = []
        self.prob = 0.0

    def fit(self, X, y):

        # 标签二值化
        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)  # 将y的标签转化为二值，便于矩阵操作

        self.classes = labelbin.classes_  # 统计总的类别
        Y = Y.astype(np.float64)  # 转为矩阵形式，方便与特征矩阵拟合

        # 转换成二分类问题
        # 特征二值化,threshold阈值根据自己的需要适当修改
        X = binarize(X, threshold=self.threshold)
        # print(X)
        feature_count = np.dot(Y.T, X)  # 矩阵转置，对相同特征进行融合
        # 因为Y是横列，想要融合特征向量（横列）,dot就是矩阵乘法，这样乘出来才能记录特征
        class_count = Y.sum(axis=0)  # 统计每一类别出现的个数 axis=0按列相加

        # 拉普拉斯平滑处理，解决零概率的问题
        alpha = 1.0
        smoothed_fc = feature_count + alpha
        smoothed_cc = class_count + alpha * 2
        self.prob = smoothed_fc/smoothed_cc.reshape(-1, 1)
        # reshape(-1,1)转换为一列

        return self


clf0 = Naive_Bayes(2.55).fit(X_train, y0_train)  # 0.2表示阈值
type_prediction = clf0.predict(X_test)
print("混淆矩阵为\n")
print(confusion_matrix(y0_test, type_prediction))
print("\n")
print(classification_report(y0_test, type_prediction))
endtime = datetime.datetime.now()

print("程序用时:{}".format(endtime - starttime))
# 输出前10张图片及预测结果


def weapon_name(num):
    numbers = {
        0: "该武器为战舰",
        1: "该武器为飞机",
        2: "该武器为导弹",
        3: "该武器为坦克"

    }

    return numbers.get(num, None)


def picprediction_example(X0_test):
    # 读取图像
    print(X0_test[k])
    image = cv2.imread(X0_test[k])
    # 显示图像
    print(weapon_name(type_prediction[k]))
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 1


print("展示测试集前10张图片的预测结果")
k = 0
while(k < 10):
    picprediction_example(X0_test)
    k = k + 1

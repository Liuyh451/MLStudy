import time
import numpy
import math
import random
from sklearn import metrics

def Get_Precision_score(y_true, y_pred):
    precision = metrics.precision_score(y_true, y_pred)
    return precision

def Get_Recall(y_true, y_pred):
    Recall = metrics.recall_score(y_true, y_pred)
    return Recall

def Get_f1_score(y_true, y_pred):
    f1_score = metrics.f1_score(y_true, y_pred)
    return f1_score

def loadImage(fileName):
    # 数据及标记数组
    dataArr = []
    labelArr = []
    csv = open(fileName)
    for line in csv.readlines():
        # 获取当前行，并按“，”切割成字段放入列表中
        curLine = line.strip().split(',')
        dataArr.append([int(num) / 255 for num in curLine[1:]])
        # 将标记信息放入标记集中
        if int(curLine[0]) == 0:
            labelArr.append(1)
        else:
            labelArr.append(-1)

    return dataArr, labelArr


class SVM:

    def __init__(self, trainDataList, trainLabelList, gaoshi=10, C=200, toler=0.001):
        '''
        SVM相关参数初始化
        '''
        self.trainDataMat = numpy.mat(trainDataList)  # 训练数据集,求得矩阵行向量
        self.trainLabelMat = numpy.mat(trainLabelList).T  # 训练标签集，转置，变为列向量
        # 查看数组的维数，m：训练集数量    n：样本特征数目
        self.m, self.n = numpy.shape(self.trainDataMat)
        self.gaoshi = gaoshi  # 高斯核分母中的σ
        self.C = C  # 惩罚参数,软间隔SVM中的平衡变量
        self.toler = toler  # 松弛变量,浮点数数值计算中允许的误差范围
        self.k = self.calcKernel()  # 核函数（初始化时提前计算）
        self.b = 0  # SVM中的偏置b
        self.alpha = [0] * self.trainDataMat.shape[0]   # α 长度为训练集数目
        self.E = [0 * self.trainLabelMat[i, 0]
                  for i in range(self.trainLabelMat.shape[0])]  # SMO运算过程中的Ei
        self.svmIndex = []  # 支持向量索引

    def calcKernel(self):
        #k[i][j] = Xi * Xj
        k = [[0 for i in range(self.m)] for j in range(self.m)]
        for i in range(self.m):
            X = self.trainDataMat[i, :]
            for j in range(i, self.m):
                Z = self.trainDataMat[j, :]
                # 先计算||X - Z||^2
                result = (X - Z) * (X - Z).T
                # 分子除以分母后去指数，得到的即为高斯核结果
                result = numpy.exp(-1 * result / (2 * self.gaoshi**2))
                # 将Xi*Xj的结果存放入k[i][j]和k[j][i]中
                k[i][j] = result
                k[j][i] = result
        # 返回高斯核矩阵
        return k

    def isSatisfyKKT(self, i):
        #检查第i个拉格朗日乘子α是否满足KKT条件
        gxi = self.calc_gxi(i)
        yi = self.trainLabelMat[i]

        if (math.fabs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            return True
        elif (math.fabs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) \
                and (math.fabs(yi * gxi - 1) < self.toler):
            return True

        return False

    def calc_gxi(self, i):
        # 初始化g(xi)
        gxi = 0
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        # 遍历每一个非零α，i为非零α的下标
        for j in index:
            # 计算g(xi)
            gxi += self.alpha[j] * self.trainLabelMat[j] * self.k[j][i]
        # 求和结束后再单独加上偏置b
        gxi += self.b
        return gxi

    def calcEi(self, i):
        # 计算g(xi)
        gxi = self.calc_gxi(i)
        #Ei = g(xi) - yi,直接将结果作为Ei返回
        return gxi - self.trainLabelMat[i]

    def getAlphaJ(self, E1, i):
        #SMO算法中在确定第一个变量的情况下, 选取第二个变量

        E2, maxE1_E2, maxIndex = 0, -1, -1  # 初始化
        nonzeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]  # 获得所有非零Ei的下标
        for j in nonzeroE:
            E2_temp = self.calcEi(j)  # 计算当前选择下标的Ei
            if math.fabs(E1 - E2_temp) > maxE1_E2:  # 发现更大的|E1-E2|
                maxE1_E2 = math.fabs(E1 - E2_temp)
                E2 = E2_temp
                maxIndex = j
        if maxIndex == -1:  # nonzeroE为空, 一开始即是空集的情况, 随机random一个
            maxIndex = i
            while maxIndex == i:  # 要保证抽到的随机数与第一个变量i不一样
                maxIndex = int(random.uniform(0, self.m))
            E2 = self.calcEi(maxIndex)
        return E2, maxIndex

    def train(self, iter=50):
        # iterStep：迭代次数，超过设置次数还未收敛则强制停止
        # parameterChanged：单次迭代中有参数改变则增加1
        iterStep = 0
        parameterChanged = 1
        while (iterStep < iter) and (parameterChanged > 0):
            print('迭代:%d:%d' % (iterStep, iter))
            iterStep += 1   # 迭代步数加1
            # 新的一轮将参数改变标志位重新置0
            parameterChanged = 0
            for i in range(self.m):
                # 查看第一个遍历是否满足KKT条件，如果不满足则作为SMO中第一个变量从而进行优化
                if self.isSatisfyKKT(i) == False:
                    # 如果下标为i的α不满足KKT条件，则进行优化
                    # 选择变量2。由于变量2的选择中涉及到|E1 - E2|，因此先计算E1
                    E1 = self.calcEi(i)

                    # 选择第2个变量
                    E2, j = self.getAlphaJ(E1, i)
                    # 获得两个变量的标签
                    y1 = self.trainLabelMat[i]
                    y2 = self.trainLabelMat[j]
                    # 复制α值作为old值
                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]
                    # 依据标签是否一致来生成不同的L和H
                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_2 + alphaOld_1)
                    # 如果两者相等，说明该变量无法再优化，直接跳到下一次循环
                    if L == H:
                        continue
                    # 计算α的新值
                    # 先获得几个k值，用来计算分母η
                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]
                    # 更新α2，该α2还未经剪切
                    alphaNew_2 = alphaOld_2 + y2 * \
                        (E1 - E2) / (k11 + k22 - 2 * k12)
                    # 剪切α2
                    if alphaNew_2 < L:
                        alphaNew_2 = L
                    elif alphaNew_2 > H:
                        alphaNew_2 = H
                    # 更新α1
                    alphaNew_1 = alphaOld_1 + y1 * \
                        y2 * (alphaOld_2 - alphaNew_2)
                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k21 * (alphaNew_2 - alphaOld_2) + self.b
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k22 * (alphaNew_2 - alphaOld_2) + self.b

                    # 依据α1和α2的值范围确定新b
                    if (alphaNew_1 > 0) and (alphaNew_1 < self.C):
                        bNew = b1New
                    elif (alphaNew_2 > 0) and (alphaNew_2 < self.C):
                        bNew = b2New
                    else:
                        bNew = (b1New + b2New) / 2

                    # 将更新后的各类值写入，进行更新
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew

                    self.E[i] = self.calcEi(i)
                    self.E[j] = self.calcEi(j)

                    # 如果α2的改变量过于小，就认为该参数未改变，不增加parameterChanged值
                    # 反之则自增1
                    if math.fabs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        parameterChanged += 1

                # 打印迭代轮数，i值，该迭代轮数修改α数目
                print("迭代: %d i:%d,  changed %d" %
                      (iterStep, i, parameterChanged))
        for i in range(self.m):
            if self.alpha[i] > 0:
                # 将支持向量的索引保存起来
                self.svmIndex.append(i)

    def calcSinglKernel(self, x1, x2):
        #单独计算核函数,根据输入的数据点x1,x2单独计算映射核函数的值, 此处用高斯核函数
        result = (x1 - x2) * (x1 - x2).T
        result = numpy.exp(-1 * result / (2 * self.gaoshi ** 2))
        # 返回结果
        return numpy.exp(result)

    def predict(self, x):
        '''
        对样本的标签进行预测
        '''
        result = 0
        for i in self.svmIndex:
            # 遍历所有支持向量，计算求和式
            tmp = self.calcSinglKernel(self.trainDataMat[i, :], numpy.mat(x))
            result += self.alpha[i] * self.trainLabelMat[i] * tmp
        result += self.b
        # 使用sign函数返回预测结果
        return numpy.sign(result)

    def test(self, testDataList, testLabelList):
        errorCnt = 0
        prelabel = []
        # 遍历测试集所有样本
        for i in range(len(testDataList)):
            print('test:%d:%d' % (i, len(testDataList)))
            result = self.predict(testDataList[i])
            print("当前样本类别为", testLabelList[i], "预测类别为", int(result[0][0]))
            prelabel.append(int(result[0][0]))
            # 如果预测与标签不一致，错误计数值加一
            if result != testLabelList[i]:
                errorCnt += 1
        # 返回正确率
        return 1 - errorCnt / len(testDataList), prelabel


if __name__ == '__main__':
    start = time.time()
    print('start read transSet')    # 获取训练集及标签
    trainDataList, trainLabelList = loadImage(
        'D:/python/lyh/MachineLearning/SVM/ds_csv/train.csv')
    # 获取测试集及标签
    print('start read testSet')
    testDataList, testLabelList = loadImage(
        'D:/python/lyh/MachineLearning/SVM/ds_csv/test.csv')
    # 初始化SVM类
    print('start init SVM')
    svm = SVM(trainDataList[:160], trainLabelList[:160], 10, 200, 0.001)
    # 开始训练
    print('start to train')
    svm.train()
    print('start to test')  # 开始测试
    accuracy, prelabel = svm.test(testDataList[:60], testLabelList[:60])
    print('the accuracy is:%d' % (accuracy * 100), '%')
    #算法评估
    target_names = ['class 0', 'class 1']
    print(metrics.classification_report(
        testLabelList[:60], prelabel, target_names=target_names))
    print("Precision_score:", '%.3f'%Get_Precision_score(
        testLabelList[:60], prelabel))
    print("Recall:", '%.3f'%Get_Recall(testLabelList[:60], prelabel))
    print("f1_score",'%.3f'% Get_f1_score(testLabelList[:60], prelabel))
    # 打印时间
    print('time span:',time.time() - start)

import cv2
import os
import numpy as np
from PIL import Image
from pylab import *
import random
import datetime 
starttime = datetime.datetime.now() 
#from __future__ import division
##############################初始化各种信息
impath='D://python//lyh//photo4//' #数据集路径
m=20    #数据集的数量
n=64   #------------->设置图片大小
size=20 #拼接图片规格（每行多少个）
listx=[]#过渡数组，用来存放每类的拼接图片
X = []  # 用于存放图片的图像矩阵
A=[]
B=[]
C=[]
###############################
#图片预处理
def pretreatment(ima,n):
    ima = ima.resize((n,n),Image.ANTIALIAS) #统一图片大小
    ima=ima.convert('L')         #转化为灰度图像,参数改为RGB，就是三通道
    im=np.array(ima).flatten()  #压成一维数组
    return im

#读取图片
for i in range(m):
    image=Image.open(impath+str(i)+'.jpg')
    X.append(pretreatment(image,n))
#计算欧式距离
def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))
    # return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))
class KMeans:
    def __init__(self,centroids,newCentroids,change,standard):
        self.standard=standard
        self.centroids=centroids
        self.newCentroids=newCentroids
        self.change=change
    def ave(self,X):
        ave=[sum(e)/len(e) for e in zip(*X)]
        return ave
    #初始质心位置改变量
    def init_centroids(self,n):
        s= random.sample(range(1,m),1)
        self.standard.append(X[s[0]])
        self.centroids= [0 for _ in range(n*n)]#生成N*N的0列表
        self.newCentroids=self.ave(self.standard)
        changed = list(map(lambda x: x[0]-x[1], zip(self.newCentroids, self.centroids)))#将列表中对应元素相减
        for i in range(len(changed)):
            self.change+=abs(changed[i])
    #更新中心坐标位置以及改变量
    def update(self):
        self.centroids=self.newCentroids
        self.newCentroids=self.ave(self.standard)
        changed = list(map(lambda x: x[0]-x[1], zip(self.newCentroids, self.centroids)))
        for i in range(len(changed)):
            self.change+=abs(changed[i])
        self.standard.clear()
test1=KMeans([],[],0,[])
test2=KMeans([],[],0,[])
test3=KMeans([],[],0,[])
test1.init_centroids(n)
test2.init_centroids(n)
test3.init_centroids(n)
cnt=0
while(test1.change!=0 and test2.change!=0 and test3.change!=0):#当change该变量为0时停止迭代
    cnt+=1
    if(cnt>5):
        break
    #用于记录每一类的图片，便于还原
    class1=[]
    class2=[]
    class3=[]
    f=-1
    for i in X:#遍历图片集
        x = i
        #计算每一类的中心坐标
        y =test1.newCentroids
        z =test2.newCentroids
        w =test3.newCentroids
        f+=1
        #由欧氏距离判断分类
        if(eucliDist(x,y)<eucliDist(x,z) and eucliDist(x,y)<eucliDist(x,w)):
            test1.standard.append(i)
            class1.append(f)
            #将距离1类近的图加入到1簇并记录图的编号
        if(eucliDist(x,z)<eucliDist(x,y) and eucliDist(x,z)<eucliDist(x,w)):
            test2.standard.append(i)
            class2.append(f)
            #将距离2类近的图加入到2簇并记录图的编号
        if(eucliDist(x,w)<eucliDist(x,y) and eucliDist(x,w)<eucliDist(x,z)):
            test3.standard.append(i)
            class3.append(f)
            #将距离3类近的图加入到3簇并记录图的编号
    test1.update()
    test2.update()
    test3.update()
# print(class1,class2,class3)
#########图片的还原以及结果展示##########
#图片拼接函数，对每一簇的图片进行拼接
def pic_fill():#为了凑齐size*size图片矩阵，填充空白图片
    path1=str(impath+'999.jpg')
    image=cv2.imread(path1)
    ima = cv2.resize(image, (64, 64),
                            interpolation=cv2.INTER_CUBIC)
    im=np.array(ima)
    return im
def pic_stitch(classx,listx,listy,name,size):
    cnt=1
    for i in classx:
        path=str(impath+str(i)+'.jpg')#遍历簇内图片
        image=cv2.imread(path)
        ima = cv2.resize(image, (64, 64),
                            interpolation=cv2.INTER_CUBIC)
        im=np.array(ima)
        listx.append(im)
        if(cnt%size==0):#每size个一行
            htitch= np.hstack(listx)
            listy.append(htitch)##实现行拼接
            listx.clear()
        cnt+=1
    if(len(classx)%size!=0): #如果不是size的倍数，则需要填充
        fillcount=size-len(classx)%size #填充数量
        for i in range(fillcount):
            listx.append(pic_fill())
        htitch= np.hstack(listx)
        listy.append(htitch)
    listx.clear()
    htitch=np.vstack(listy) #实现列拼接
    name=str(name)
    cv2.imshow(name,htitch)#生成图片
    cv2.waitKey(0)
    cv2.destroyAllWindows()

pic_stitch(class1,listx,A,1,size)
pic_stitch(class2,listx,B,2,size)
pic_stitch(class3,listx,C,3,size)
endtime = datetime.datetime.now()
print ("程序用时:{}".format(endtime - starttime))





    


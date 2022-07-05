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
impath='D://python//lyh//photo3//' #数据集路径
m=50    #数据集的数量
n=256   #------------->设置图片大小
size=20 #拼接图片规格（每行多少个）
listx=[]#过渡数组，用来存放每类的拼接图片
X = []  # 用于存放图片的图像矩阵
A=[]
B=[]
C=[]
#确定最初中心点坐标   
standard_1=[]
standard_2=[]
standard_3=[]
###############################
#图片预处理
def pretreatment(ima,n):
    ima = ima.resize((n,n),Image.ANTIALIAS) #统一图片大小
    ima=ima.convert('RGB')         #转化为灰度图像,参数改为RGB，就是三通道
    im=np.array(ima).flatten()  #压成一维数组
    return im

#读取图片
for i in range(m):
    image=Image.open(impath+str(i)+'.jpg')
    X.append(pretreatment(image,n))
    
def init():    #初始化中心坐标
    #静态初始化坐标点
    # standard_1.append(X[0])
    # standard_2.append(X[1])
    # standard_3.append(X[2])
    #随机初始化坐标点
    randlist = random.sample(range(1,m), 3)    
    standard_1.append(X[randlist[0]])
    standard_2.append(X[randlist[1]])
    standard_3.append(X[randlist[2]])

init()

#计算欧式距离
def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))
    # return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))
    
#计算当前聚类簇的中心坐标
def ave(X):
    ave=[sum(e)/len(e) for e in zip(*X)]
    return ave
#初始质心位置改变量
def init_centroids(n,standard):
    centroids= [0 for _ in range(n*n)]#生成N*N的0列表
    newCentroids=ave(standard)
    change = list(map(lambda x: x[0]-x[1], zip(newCentroids, centroids)))#将列表中对应元素相减
    return centroids,newCentroids,change
#更新中心坐标位置以及改变量
def update(standard,newCentroids):
        centroids=newCentroids
        newCentroids=ave(standard)
        change = list(map(lambda x: x[0]-x[1], zip(newCentroids, centroids)))
        standard.clear()
        return centroids,newCentroids,change
    
centroids1,newCentroids1,changed1=init_centroids(n,standard_1)
centroids2,newCentroids2,changed2=init_centroids(n,standard_2)
centroids3,newCentroids3,changed3=init_centroids(n,standard_3)
#限制迭代过程
def func(change):
    sum=0
    for i in range(len(change)):
        sum+=abs(change[i])
    return sum
###################---Kmeans 迭代--- #################
flag=True
while(flag):
    #当change该变量为0时停止迭代
    #用于记录每一类的图片，便于还原
    class1=[]
    class2=[]
    class3=[]
    f=-1
    for i in X:#遍历图片集
        x = i
        #计算每一类的中心坐标
        y =newCentroids1
        z =newCentroids2
        w =newCentroids3
        f+=1
        #由欧氏距离判断分类
        if(eucliDist(x,y)<eucliDist(x,z) and eucliDist(x,y)<eucliDist(x,w)):
            standard_1.append(i)
            class1.append(f)
            #将距离1类近的图加入到1簇并记录图的编号
        if(eucliDist(x,z)<eucliDist(x,y) and eucliDist(x,z)<eucliDist(x,w)):
            standard_2.append(i)
            class2.append(f)
            #将距离2类近的图加入到2簇并记录图的编号
        if(eucliDist(x,w)<eucliDist(x,y) and eucliDist(x,w)<eucliDist(x,z)):
            standard_3.append(i)
            class3.append(f)
            #将距离3类近的图加入到3簇并记录图的编号
            
    #每轮遍历结束后更新中心坐标及change改变量
    print("各类的图片数量为：")
    print(len(standard_1),"---",len(standard_2),"---",len(standard_3)) 
    centroids1,newCentroids1,changed1=update(standard_1,newCentroids1)
    centroids2,newCentroids2,changed2=update(standard_2,newCentroids2) 
    centroids3,newCentroids3,changed3=update(standard_3,newCentroids3)
    #更新flag的值，当且仅当三个改变量同时为0时flag为flase停止迭代
    flag=func(changed1)==0 and func(changed2)==0 and func(changed3)==0
    flag=not flag
    print("本轮迭代聚类结果为：")#打印出每轮的类
    print(class1,class2,class3)
print("最终聚类结果为：")
print(class1,class2,class3)
#########图片的还原以及结果展示##########
#图片拼接函数，对每一簇的图片进行拼接
def pic_fill():#为了凑齐size*size图片矩阵，填充空白图片
    path1=str(impath+'999.jpg')
    image=cv2.imread(path1)
    ima = cv2.resize(image, (128, 128),
                            interpolation=cv2.INTER_CUBIC)
    im=np.array(ima)
    return im
def pic_stitch(classx,listx,listy,name,size):
    cnt=1
    for i in classx:
        path=str(impath+str(i)+'.jpg')#遍历簇内图片
        image=cv2.imread(path)
        ima = cv2.resize(image, (128, 128),
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





    


import numpy as np
import cv2
import csv,os
def sys_moments(img):
    '''
    opencv_python自带求矩以及不变矩的函数
    :param img: 灰度图像，对于二值图像来说就只有两个灰度0和255
    :return: 返回以10为底对数化后的hu不变矩
    '''
    moments = cv2.moments(img)  # 返回的是一个字典，三阶及以下的几何矩（mpq）、中心矩(mupq)和归一化的矩(nupq)
    humoments = cv2.HuMoments(moments)  # 根据几何矩（mpq）、中心矩(mupq)和归一化的矩(nupq)计算出hu不变矩
    # 因为直接计算出来的矩可能很小或者很大，因此取对数好比较,这里的对数底数为e,通过对数除法的性质将其转换为以10为底的对数
    humoment = (np.log(np.abs(humoments))) / np.log(10)
    return humoment


def def_moments(img_gray):
    '''
    自定义求矩函数，主要是根据公式将一个个参数求出
    :param img_gray:  灰度图像，对于二值图像来说就只有两个灰度0和255
    :return: 返回以10为底对数化后的hu不变矩
    '''
    '''
        由于7个不变矩的变化范围很大,为了便于比较,可利用取对数的方法进行数据压缩;
        同时考虑到不变矩有可能出现负值的情况,因此,在取对数之前先取绝对值
        经修正后的不变矩特征具有平移 、旋转和比例不变性
    '''
    # 标准矩定义为m_pq = sumsum(x^p * y^q * f(x, y))其中f(x,y)为像素点处的灰度值
    row, col = img_gray.shape
    # 计算图像的0阶几何矩
    m00 = img_gray.sum()
    ##初始化一到三阶几何矩
    # 计算一阶矩阵
    m10 = m01 = 0
    # 计算图像的二阶、三阶几何矩
    m11 = m20 = m02 = m12 = m21 = m30 = m03 = 0
    for i in range(row):
        m10 += (i * img_gray[i]).sum()  # sum表示将一行的灰度值进行相加
        m20 += (i ** 2 * img_gray[i]).sum()
        m30 += (i ** 3 * img_gray[i]).sum()
        for j in range(col):
            m11 += i * j * img_gray[i][j]
            m12 += i * j ** 2 * img_gray[i][j]
            m21 += i ** 2 * j * img_gray[i][j]
    for j in range(col):
        m01 += (j * img_gray[:, j]).sum()
        m02 += (j ** 2 * img_gray[:, j]).sum()
        m30 += (j ** 3 * img_gray[:, j]).sum()
    # 由标准矩我们可以得到图像的"重心"
    u10 = m10 / m00
    u01 = m01 / m00
    # 计算图像的二阶中心矩、三阶中心矩
    y00 = m00
    y10 = y01 = 0
    y11 = m11 - u01 * m10
    y20 = m20 - u10 * m10
    y02 = m02 - u01 * m01
    y30 = m30 - 3 * u10 * m20 + 2 * u10 ** 2 * m10
    y12 = m12 - 2 * u01 * m11 - u10 * m02 + 2 * u01 ** 2 * m10
    y21 = m21 - 2 * u10 * m11 - u01 * m20 + 2 * u10 ** 2 * m01
    y03 = m03 - 3 * u01 * m02 + 2 * u01 ** 2 * m01
    # 计算图像的归一化中心矩
    n20 = y20 / m00 ** 2
    n02 = y02 / m00 ** 2
    n11 = y11 / m00 ** 2
    n30 = y30 / m00 ** 2.5
    n03 = y03 / m00 ** 2.5
    n12 = y12 / m00 ** 2.5
    n21 = y21 / m00 ** 2.5
    # 计算图像的七个不变矩
    h1 = n20 + n02
    h2 = (n20 - n02) ** 2 + 4 * n11 ** 2
    h3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h5 = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + (3 * n21 - n03) * (n21 + n03) \
         * (3 * (n30 + n12) ** 2 - (
            n21 + n03) ** 2)
    h6 = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03)
    h7 = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + (3 * n12 - n30) * (n21 + n03) \
         * (3 * (n30 + n12) ** 2 - (
            n21 + n03) ** 2)
    inv_m7 = [h1, h2, h3, h4, h5, h6, h7]
    humoments = np.log(np.abs(inv_m7))
    return humoments

def main():
    img = cv2.imread('F:\\0\\0.jpg', 0)
    sys_humoments = sys_moments(img)
    #def_humoments = sys_moments(img)
    #print('Hu不变矩为：\n', sys_humoments)
    #print('自定义函数：\n', def_humoments)
    return sys_humoments

def convert_img_to_csv(img_dir):
    #设置需要保存的csv路径
    with open(r"F:\pth\test.csv","w",newline="") as f:
        #设置csv文件的列名
        column_name = ["label"]
        column_name.extend(["pixel%d"%i for i in range(7)])
        #将列名写入到csv文件中
        writer = csv.writer(f)
        writer.writerow(column_name)
        for i in range(3):
            #获取目录的路径
            img_temp_dir = os.path.join(img_dir,str(i))
            #获取该目录下所有的文件
            img_list = os.listdir(img_temp_dir)
            #遍历所有的文件名称
            for img_name in img_list:
                #判断文件是否为目录,如果为目录则不处理
                if not os.path.isdir(img_name):
                    #获取图片的路径
                    img_path = os.path.join(img_temp_dir,img_name)
                    #因为图片是黑白的，所以以灰色读取图片
                    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                    sys_humoments = sys_moments(img)
                    #图片标签
                    row_data = [i]
                    #获取图片的像素
                    row_data.extend(sys_humoments.flatten())
                    #将图片数据写入到csv文件中
                    writer.writerow(row_data)

if __name__ == '__main__':
    main()
    convert_img_to_csv(r"F:\pp")

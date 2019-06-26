import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as io


# p阶AR模型参数与预测值计算
def ar_least_square(sample,p):
    matrix_x=np.zeros((sample.size-p,p))
    matrix_x=np.matrix(matrix_x)
    array=sample.reshape(sample.size)
    j=0
    for i in range(0,sample.size-p):
	    matrix_x[i,0:p]=array[j:j+p]
	    j=j+1
    matrix_y=np.array(array[p:sample.size])
    matrix_y=matrix_y.reshape(sample.size-p,1)
    matrix_y=np.matrix(matrix_y)
    fi=np.dot(np.dot((np.dot(matrix_x.T,matrix_x)).I,matrix_x.T),matrix_y)
    matrix_y=np.dot(matrix_x,fi)
    matrix_y=np.row_stack((array[0:p].reshape(p,1),matrix_y))
    return fi,matrix_y


# 计算AIC
def ar_aic(rss, p):
    n = rss.size;
    s2 = np.var(rss);
    return 2 * p + n * math.log(s2);


# 计算SC
def ar_sc(rss, p):
    n = rss.size;
    s2 = np.var(rss);
    return p * math.log(n) + n * math.log(s2);


if __name__ == '__main__':
    fileName = "105"
    data = io.loadmat("C:\E\PythonProjects\data\\" + fileName + ".mat")
    dataName = ["X105_BA_time", "X105_DE_time", "X105_FE_time"]
    titleNum=1
    dataLength=1000
    num=1
    dataTest = np.array(data[dataName[titleNum]]).reshape(1, -1)[0]
    t = np.linspace(dataLength * (num - 1), dataLength * num, dataLength)
    measurements = dataTest[dataLength * (num - 1):dataLength * num]
    t = np.linspace(0,len(measurements),len(measurements))

    # p=1
    # aics=[]
    # scs=[]
    # while p<len(measurements)-1:
    #     print("定阶%d...",p)
    #     fi, rate_predict=ar_least_square(measurements,p)
    #     rss=rate_predict-measurements
    #     aics.append(ar_aic(rss,p))
    #     scs.append(ar_sc(rss,p))
    #     p+=1
    # t = np.linspace(1, p-1, p-1)
    # plt.figure(figsize=(12, 6))
    # plt.plot(t, aics,color='g',label="AICS")
    # plt.xlabel("p")
    # plt.legend()
    # plt.show()
    # plt.figure(figsize=(12, 6))
    # plt.plot(t, scs,color='b',label="SCS")
    # plt.xlabel("p")
    # plt.legend()
    # plt.show()

    fi_p, rate_predict_p = ar_least_square(measurements,100)
    plt.figure(figsize=(8, 4))
    plt.plot(t, measurements, color='b', label="Original")
    plt.plot(t, rate_predict_p, color='g', label="After Filter")
    plt.xlabel("Time t/ms")
    plt.ylabel("Amplitude A/mv")
    plt.legend()
    plt.savefig('AR')
    plt.show()
    # io.savemat('C:\E\PythonProjects\data\\' + fileName + "a.mat",{dataName[1]: rate_predict_p})
from PyEMD import EEMD
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from scipy.stats import pearsonr
import pandas as pd
from pykalman import KalmanFilter
from numpy import *
import os


if __name__ == '__main__':
    fileName="105"
    data = io.loadmat("C:\E\PythonProjects\data\\"+fileName+".mat")
    dataName=["X105_BA_time","X105_DE_time","X105_FE_time"]
    # dataName = ["X100_DE_time", "X100_FE_time"]
    num = 1
    while num <= 1:
        titleNum = 1
        dataLength=1000
        r = []
        while titleNum < 2:
            print("Load Data File......")
            dataTest = np.array(data[dataName[titleNum]]).reshape(1, -1)[0]
            t = np.linspace(dataLength*(num-1),dataLength*num, dataLength)
            measurements=dataTest[dataLength*(num-1):dataLength*num]

            print("IMFs......")
            IMFs=[]
            IMFs = EEMD().eemd(measurements,t)
            N = IMFs.shape[0]+1

            plt.figure(figsize=(8, 16))
            plt.subplot(N,1,1)
            plt.plot(t, measurements, 'b')
            plt.title("Original")

            cors=[]
            kurs=[]
            print("cors......")
            for n, imf in enumerate(IMFs):
                cors.append(pearsonr(measurements,imf)[0])
                plt.subplot(N,1,n+2)
                plt.plot(t, imf, 'g')
                plt.title("IMF "+str(n+1))
            print("cors:")
            print(cors)

            corThreshold=max(cors)*0.1
            print("corThreshold:%lf"%corThreshold)
            corCanceleds=[]
            for n, imf in enumerate(IMFs):
                if(cors[n]>=corThreshold):
                    corCanceleds.append(cors[n])
                    s = pd.Series(imf)
                    kurs.append(s.kurt())
            print("corCanceleds:")
            print(corCanceleds)
            print("kurs:")
            print(kurs)

            maxKurIndex=kurs.index(max(kurs))
            print("maxKurIndex:%d" % maxKurIndex)
            secondKurIndex=kurs.index(min(kurs))
            for n, kur in enumerate(kurs):
                if kur>=kurs[secondKurIndex] and kur<max(kurs):
                    secondKurIndex=n
            print("secondKurIndex:%d"%secondKurIndex)

            if maxKurIndex==secondKurIndex:
                r.append(IMFs[maxKurIndex])
            else:  r.append(IMFs[maxKurIndex]+IMFs[secondKurIndex])
            plt.subplot(N, 1, 1)
            plt.tight_layout()
            plt.show()
            titleNum+=1
        io.savemat('C:\E\PythonProjects\data\\' + fileName + "_%d" % num + ".mat", {dataName[1]: r[0].reshape(-1, 1)})
        plt.figure(figsize=(8, 4))
        plt.plot(t, measurements, 'b',label="Original")
        plt.plot(t, r[0], 'g',label="After Filter")
        plt.xlabel("Time t/ms")
        plt.ylabel("Amplitude A/mv")
        plt.legend()
        plt.savefig('fig1')
        plt.show()
        num+=1
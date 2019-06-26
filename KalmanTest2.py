import matplotlib.pyplot as plt
import scipy.io as io
from pykalman import KalmanFilter
from numpy import *


if __name__ == '__main__':
    fileName="106"
    data = io.loadmat("C:\E\PythonProjects\data\\"+fileName+".mat")
    dataName=["X105_BA_time","X106_DE_time","X105_FE_time"]
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
            print("Kalman Filter...")
            kf = KalmanFilter(n_dim_obs=1,
                              n_dim_state=1,
                              initial_state_mean=measurements[0],
                              initial_state_covariance=np.cov(measurements),
                              transition_matrices=[1],
                              transition_covariance=np.eye(1),
                              transition_offsets=None,
                              observation_matrices=[1],
                              observation_covariance=1
                              )
            test, cov = kf.filter(measurements)
            test=np.array(test).reshape(1, -1)[0]
            plt.figure(figsize=(8, 4))
            plt.plot(t, measurements, color='b', label="Original")
            plt.plot(t, test, color='g', label="After Filter")
            plt.xlabel("Time t/ms")
            plt.ylabel("Amplitude A/mv")
            plt.legend()
            plt.savefig('KALMAN')
            titleNum+=1

        io.savemat('C:\E\PythonProjects\data\\' + fileName + "k.mat", {dataName[1]: test.reshape(-1, 1)})
        num+=1
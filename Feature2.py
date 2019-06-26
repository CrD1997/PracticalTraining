import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io as scio
import numpy as np
import pandas as pd
from random import sample
from random import shuffle
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as io


def dataWithNoise(x,snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    x = np.array(x)
    noise = np.array(noise)
    x_with_noise = x + noise
    return x_with_noise


# 计算四个特征值
def get4Features(x):
    # 计算标准差
    n = len(x)
    if n <= 1:
        std_deviation = 0
        rms = 0
        crest_factor = 0
    else:
        std_deviation = np.sqrt((n * np.sum(x ** 2) - (np.sum(x) ** 2)) / (n * (n - 1)))
        rms = np.sqrt(np.sum(x ** 2) / n)
        crest_factor = max(x)/rms
    s = pd.Series(x)
    skewness = s.skew()
    kurtosis = s.kurt()

    features = {}
    features["StdDeviation"] = std_deviation
    features["Skewness"] = skewness
    features["Kurtosis"] = kurtosis
    features["CrestFactor"] = crest_factor

    return features


def loadData(fileNum, hz):
    fileName = fileNum + "h.mat"
    if int(fileNum) < 100:
        data = scio.loadmat(fileName)['X0'+fileNum+'_DE_time'].reshape(-1)
    else:
        data = scio.loadmat(fileName)['X' + fileNum + '_DE_time'].reshape(-1)
    return_data = data[:90000]
    time = []
    for i in range(len(return_data)):
        time.append((1 / hz) * i)  # hz为数据的采样频率
    return return_data, time


if __name__ == '__main__':
    mpl.rcParams['agg.path.chunksize'] = 10000
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 加载数据
    print("加载数据...")
    normal_data, time_normal = loadData('98', 12000)
    inner_data, time_inner = loadData('106', 12000)
    ball_data, time_ball = loadData('119', 12000)
    outer_data, time_outer = loadData('131', 12000)

    # 为原始数据添加噪声
    print("添加噪声...")
    normal_data_with_noise = dataWithNoise(normal_data, 10)
    inner_data_with_noise = dataWithNoise(inner_data, 10)
    ball_data_with_noise = dataWithNoise(ball_data, 10)
    outer_data_with_noise = dataWithNoise(outer_data, 10)

    # 绘制原始与添加白噪声之后的数据
    print("绘制原始与添加白噪声之后的数据...")
    plt.figure(figsize=(16, 8))
    plt.plot(time_normal, normal_data, color='b', label="Normal")
    plt.plot(time_normal, normal_data_with_noise, color='r', label="Normal With Noise")
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.plot(time_inner, inner_data, color='b', label="Inner")
    plt.plot(time_inner, inner_data_with_noise, color='r', label="Inner With Noise")
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.plot(time_ball, ball_data, color='b', label="Ball")
    plt.plot(time_ball, ball_data_with_noise, color='r', label="Ball With Noise")
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.plot(time_outer, outer_data, color='b', label="Outer")
    plt.plot(time_outer, outer_data_with_noise, color='r', label="Outer With Noise")
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.show()

    # 绘制特征值散点图
    print("绘制特征值散点图...")
    normal_std_deviation = []
    normal_skewness = []
    normal_kurtosis = []
    normal_crest_factor = []
    normal_y = []
    for i in range(0,300):
        x=np.array(sample(list(normal_data), 800))
        features=get4Features(x)
        normal_std_deviation.append(features["StdDeviation"])
        normal_skewness.append(features["Skewness"])
        normal_kurtosis.append(features["Kurtosis"])
        normal_crest_factor.append(features["CrestFactor"])
        normal_y.append(1)
    inner_std_deviation = []
    inner_skewness = []
    inner_kurtosis = []
    inner_crest_factor = []
    inner_y = []
    for i in range(0, 300):
        x = np.array(sample(list(inner_data), 800))
        features = get4Features(x)
        inner_std_deviation.append(features["StdDeviation"])
        inner_skewness.append(features["Skewness"])
        inner_kurtosis.append(features["Kurtosis"])
        inner_crest_factor.append(features["CrestFactor"])
        inner_y.append(2)
    ball_std_deviation = []
    ball_skewness = []
    ball_kurtosis = []
    ball_crest_factor = []
    ball_y = []
    for i in range(0, 300):
        x = np.array(sample(list(ball_data), 800))
        features = get4Features(x)
        ball_std_deviation.append(features["StdDeviation"])
        ball_skewness.append(features["Skewness"])
        ball_kurtosis.append(features["Kurtosis"])
        ball_crest_factor.append(features["CrestFactor"])
        ball_y.append(3)
    outer_std_deviation = []
    outer_skewness = []
    outer_kurtosis = []
    outer_crest_factor = []
    outer_y = []
    for i in range(0, 300):
        x = np.array(sample(list(outer_data), 800))
        features = get4Features(x)
        outer_std_deviation.append(features["StdDeviation"])
        outer_skewness.append(features["Skewness"])
        outer_kurtosis.append(features["Kurtosis"])
        outer_crest_factor.append(features["CrestFactor"])
        outer_y.append(4)
    # 合并数据
    print("合并数据...")
    std_deviation = np.hstack((normal_std_deviation,inner_std_deviation,ball_std_deviation,outer_std_deviation))
    skewness = np.hstack((normal_skewness,inner_skewness,ball_skewness,outer_skewness))
    kurtosis = np.hstack((normal_kurtosis,inner_kurtosis,ball_kurtosis,outer_kurtosis))
    crest_factor = np.hstack((normal_crest_factor,inner_crest_factor,ball_crest_factor,outer_crest_factor))
    y = np.hstack((normal_y,inner_y,ball_y,outer_y))
    orignal_data = np.vstack((std_deviation,skewness,kurtosis,crest_factor,y))
    final_data = np.transpose(np.mat(orignal_data)).tolist()
    shuffle(final_data)
    np.savetxt('data_h1.csv', np.mat(final_data), delimiter = ',')

    print("绘制偏度散点图...")
    x = [i for i in range(len(normal_skewness))]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, normal_skewness, c='r', label="Normal", marker='+')
    ax1.scatter(x, inner_skewness, c='b', label="Inner Raceway Fault", marker='+')
    ax1.scatter(x, ball_skewness, c='y', label="Rolling Element Fault", marker='+')
    ax1.scatter(x, outer_skewness, c='g', label="Outer Raceway Fault", marker='+')
    plt.xlabel("Smaple")
    plt.ylabel("Skewness")
    plt.legend()
    plt.savefig("Skewness")
    plt.show()

    print("绘制峰度散点图...")
    x = [i for i in range(len(normal_kurtosis))]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, normal_kurtosis, c='r', label="Normal", marker='+')
    ax1.scatter(x, inner_kurtosis, c='b', label="Inner Raceway Fault", marker='+')
    ax1.scatter(x, ball_kurtosis, c='y', label="Rolling Element Fault", marker='+')
    ax1.scatter(x, outer_kurtosis, c='g', label="Outer Raceway Fault", marker='+')
    plt.xlabel("Smaple")
    plt.ylabel("Kurtosis")
    plt.legend()
    plt.savefig("Kurtosis")
    plt.show()

    print("绘制波峰因子散点图...")
    x = [i for i in range(len(normal_crest_factor))]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, normal_crest_factor, c='r', label="Normal", marker='+')
    ax1.scatter(x, inner_crest_factor, c='b', label="Inner Raceway Fault", marker='+')
    ax1.scatter(x, ball_crest_factor, c='y', label="Rolling Element Fault", marker='+')
    ax1.scatter(x, outer_crest_factor, c='g', label="Outer Raceway Fault", marker='+')
    plt.xlabel("Smaple")
    plt.ylabel("Crest Factor")
    plt.legend()
    plt.savefig("Crest Factor")
    plt.show()

    print("绘制标准差散点图...")
    x = [i for i in range(len(normal_std_deviation))]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, normal_std_deviation, c='r', label="Normal", marker='+')
    ax1.scatter(x, inner_std_deviation, c='b', label="Inner Raceway Fault", marker='+')
    ax1.scatter(x, ball_std_deviation, c='y', label="Rolling Element Fault", marker='+')
    ax1.scatter(x, outer_std_deviation, c='g', label="Outer Raceway Fault", marker='+')
    plt.xlabel("Smaple")
    plt.ylabel("Standard Deviation")
    plt.legend()
    plt.savefig("StandardDeviation")
    plt.show()

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(normal_kurtosis, normal_skewness, normal_crest_factor, c='r', label="Normal",marker='+')
    ax.scatter(inner_kurtosis,inner_skewness,inner_crest_factor, c='b', label="Inner Raceway Fault", marker='+')
    ax.scatter(ball_kurtosis, ball_skewness, ball_crest_factor, c='y', label="Rolling Element Fault", marker='+')
    ax.scatter(outer_kurtosis,outer_skewness, outer_crest_factor, c='g', label="Outer Raceway Fault", marker='+')
    ax.set_xlabel('Kurtosis')
    ax.set_ylabel('Skewness')
    ax.set_zlabel('Crest Factor')
    plt.savefig('3D')
    plt.show()
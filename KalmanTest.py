import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

# kf = KalmanFilter(n_dim_obs=1,
#                   n_dim_state=1,
#                   initial_state_mean=23,
#                   initial_state_covariance=5,
#                   transition_matrices=[1],
#                   observation_matrices=[1],
#                   observation_covariance=4,
#                   transition_covariance=np.eye(1),
#                   transition_offsets=None)
#
# actual = [23]*100
# sim = actual + np.random.normal(0,1,100)
# state_means, state_covariance = kf.filter(sim)

# plt.plot(actual,'r-')
# plt.plot(sim,'k-')
# plt.plot(state_means,'g-')
# plt.show()

# import scipy.io as io
# kf = KalmanFilter(n_dim_obs=1,
#                   n_dim_state=1,
#                   initial_state_mean=0,
#                   initial_state_covariance=0,
#                   transition_matrices=[1],
#                   observation_matrices=[1],
#                   observation_covariance=0,
#                   transition_covariance=np.eye(1),
#                   transition_offsets=None)
#
# actual = [0]*2000
# data = io.loadmat("C:\E\PythonProjects\data\\97.mat")
# test = np.array(data["X097_DE_time"]).reshape(1, -1)[0]
# sim=test[:2000]
# state_means, state_covariance = kf.filter(sim)
#
# plt.figure(figsize=(18, 6))
# plt.plot(actual,'r-')
# plt.plot(state_means,'g-')
# plt.plot(sim,'k-')
# plt.show()

import scipy.io as io

data = io.loadmat("C:\E\PythonProjects\data\\97.mat")
test = np.array(data["X097_DE_time"]).reshape(1, -1)[0]
measurements=test[:2000]
kf = KalmanFilter(n_dim_obs = 1,
                  n_dim_state = 1,
                  initial_state_mean = measurements[0],
                  initial_state_covariance = np.cov(measurements),
                  transition_matrices = [1],
                  transition_covariance = np.eye(1),
                  transition_offsets = None,
                  observation_matrices = [1],
                  observation_covariance = 10
                 )
mean,cov = kf.filter(measurements)
plt.figure(figsize=(18, 6))
plt.plot(measurements,'g-')
plt.plot(mean, 'r-')
plt.show()
# measurements = np.asarray([[1,0], [0,0], [0,1]])  # 3 observations
# kf = kf.em(measurements)
# (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
# (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
# plt.figure(figsize=(18, 6))
# plt.plot(measurements,'k-')
# plt.plot(filtered_state_means,'g-')
# plt.show()
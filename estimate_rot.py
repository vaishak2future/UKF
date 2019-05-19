# data files are numbered on the server.
# for exmaple imuRaw1.mat, imuRaw2.mat and so on.
# write a function that takes in an input number (1 through 6)
# reads in the corresponding imu Data, and estimates
# roll pitch and yaw using an extended kalman filter
import time

import helper
import quaternion
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def read_data_imu(num):
    return sio.loadmat("./imu/imuRaw"+ str(num)+".mat")


def read_data_vicon(num):
    return sio.loadmat("./vicon/viconRot"+ str(num)+".mat")

#Q = np.eye(6)*(10**-2)
#R = np.eye(6)*(10**-1)
Q = np.eye(6) * 0.003
Q[3:,3:] = Q[3:,3:]/100000000
R = np.eye(6) * 0.005

def estimate_rot(data_num=1):
    #P = 0.00001*np.eye(6)
    P = 0.00000000001*np.eye(6)
    imu_data = read_data_imu(data_num)
    state = np.asarray([1, 0, 0, 0, 0, 0, 0])
    ts = np.squeeze(imu_data['ts'])
    real_measurement = np.squeeze(imu_data['vals']).transpose()
    rpy = []
    Y_qmean = np.asarray([1, 0, 0, 0])
    for i in range(1, ts.shape[0]):
        dt = ts[i] - ts[i - 1]
        Y,Z = helper.sigma_points(P,Q, state, dt)
        K, P, Z_mean,Y_mean, Y_qmean  = helper.find_gain(Y, Y_qmean,Z,R)
        corrected_measurements = helper.correct_measurements(real_measurement[i, :3], real_measurement[i, 3:])
        state = helper.state_update(K,Z_mean,Y_qmean,Y_mean,corrected_measurements)
        rpy.append(quaternion.toRPE(state))
    return np.asarray(rpy)[:,0], np.asarray(rpy)[:,1], np.asarray(rpy)[:,2]


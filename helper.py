import numpy as np
import quaternion
import copy

gyro_bias = [371.33, 374.76, 399.82]
acc_bias = [500, 500, 500]
acc_sens = 34.7455
gyro_sens = 2.02

def correct_measurements(acc, gyro):
    acc_new = (acc - acc_bias) * 3300 / 1023 / acc_sens
    gyro_new = (gyro - gyro_bias) * 3300 / 1023 * np.pi / 180 / gyro_sens
    return [-acc_new[0], -acc_new[1], acc_new[2], gyro_new[1], gyro_new[2], gyro_new[0]]

def process_model(states, dt):
    q = quaternion.multiply(states[:,:4], quaternion.from_rv(states[:,4:], dt))
    w = states[:,4:]
    return np.column_stack((q, w))

def acc_model(states):
    g = np.array([[0, 0, 0, 9.8]])
    inv_states = copy.deepcopy(states[:,:4])
    inv_states[:,1:4] = -inv_states[:, 1:4]
    g_prime = quaternion.multiply(quaternion.multiply(inv_states, g), states[:, :4])
    return g_prime[:,1:4]

def sensor_model(states):
    return np.column_stack((acc_model(states), states[:, 4:])).squeeze()


def state_update(K,Z_mean,Y_qmean,Y_mean,measurement):
    v = measurement - Z_mean
    kv = np.matmul(K, v)
    q = quaternion.multiply(Y_qmean[np.newaxis, :], quaternion.from_rv(kv[np.newaxis, :3], 1))
    w = Y_mean[4:] + kv[3:]
    return np.hstack((q.flatten(), w))

def sigma_points(P,Q , state, dt):
    W = np.vstack((np.sqrt(2*6 )*np.linalg.cholesky((P + Q)).transpose(),
                    -np.sqrt(2*6)*(np.linalg.cholesky((P + Q)).transpose())))
    X = np.zeros((7, 12))
    X[:4,:] = quaternion.multiply(state[np.newaxis, :4], quaternion.from_rv(W[:,:3])).transpose()
    X[4:, :] = (state[4:] + W[:,3:]).T
    X = X.T
    Y = process_model(X, dt)
    Z = sensor_model(X)
    return Y,Z


def find_mean(Y_qmean,Y,Z):
    Y_mean = np.hstack((Y_qmean.flatten(), np.mean(Y[:, 4:], axis=0)))
    Z_mean = np.mean(Z, axis=0)
    return Y_mean,Z_mean

def qmean(Y, Y_qmean):
    iter = 0
    while (1):
        iter += 1
        Y_qmean_inv = np.hstack((Y_qmean[0], -Y_qmean[1:]))
        ev_i = quaternion.toVec(quaternion.multiply(Y[:,:4], Y_qmean_inv[np.newaxis, :]))
        e_quat = quaternion.from_rv(np.mean(ev_i, axis=0)[np.newaxis, :])
        Y_qmean = quaternion.multiply(e_quat, Y_qmean[np.newaxis, :]).squeeze()
        if np.linalg.norm(np.mean(ev_i, axis=0)) < 0.001 or iter > 10000:
            break
    return Y_qmean,ev_i

def find_cov(ev_i,Y_mean,Y,Z,Z_mean,R):
    w_i = Y[:, 4:] - Y_mean[4:]
    W = np.vstack((ev_i.T, w_i.T))
    YCov = np.matmul(W, W.T)/ (2 * 6)
    Z_new = (Z - Z_mean).T
    ZCov = np.matmul(Z_new, Z_new.T)/ (2 * 6)
    Pvv = ZCov + R
    CrossCov = np.matmul(W, Z_new.transpose())/ (2 * 6)
    return YCov,ZCov,CrossCov,Pvv

def find_gain(Y, Y_qmean,Z,R):
    Y_qmean_new,ev_i = qmean(Y, Y_qmean)
    YMean, ZMean = find_mean(Y_qmean_new, Y, Z)
    YCov,ZCov,CrossCov,Pvv = find_cov(ev_i,YMean,Y,Z,ZMean,R)
    K = np.matmul(CrossCov, np.linalg.inv(Pvv))
    P = YCov - np.matmul(np.matmul(K, Pvv), K.transpose())
    return K, P, ZMean, YMean, Y_qmean_new
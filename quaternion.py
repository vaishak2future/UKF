import numpy as np


def init(v, a):
    return np.hstack((np.cos(a / 2.0)[:,np.newaxis], v * np.repeat(np.sin(a / 2.0)[:,np.newaxis], 3, axis=1)))

def from_rv(w, dt=1):
    angle = np.linalg.norm(w, axis=1) * dt
    axis = w / np.repeat(np.linalg.norm(w, axis=1)[:, np.newaxis], 3, axis=1)

    axis[np.isnan(axis[:,0]*axis[:,1]*axis[:,2])] = [0,0,0]
    return init(axis, angle)

def multiply(q1, q2):
    if q1.shape[0] == 1:
        q_1 = np.repeat(q1[:, :, np.newaxis], q2.shape[0], axis=0).squeeze(axis=2)
    else:
        q_1 = q1
    if q2.shape[0] == 1:
        q_2 = np.repeat(q2[:, :, np.newaxis], q1.shape[0], axis=0).squeeze(axis=2)
    else:
        q_2 = q2
    return np.column_stack((q_1[:,0]*q_2[:,0] - q_1[:,1]*q_2[:,1] - q_1[:,2]*q_2[:,2] - q_1[:,3]*q_2[:,3],
                            q_1[:,0]*q_2[:,1] + q_1[:,1]*q_2[:,0] + q_1[:,2]*q_2[:,3] - q_1[:,3]*q_2[:,2],
                            q_1[:,0]*q_2[:,2] - q_1[:,1]*q_2[:,3] + q_1[:,2]*q_2[:,0] + q_1[:,3]*q_2[:,1],
                            q_1[:,0]*q_2[:,3] + q_1[:,1]*q_2[:,2] - q_1[:,2]*q_2[:,1] + q_1[:,3]*q_2[:,0],))

def toRPE(q):
    sinr_cosp = +2.0 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = +1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = +2.0 * (q[0] * q[2] - q[3] * q[1])
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = +2.0 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = +1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return [roll, pitch, yaw]

def toVec(q):
    q = q/np.repeat(np.linalg.norm(q, axis=1)[:, np.newaxis], 4, axis=1)
    theta = 2 * np.arccos(q[:, 0])
    return q[:, 1:] * np.repeat(theta[:,np.newaxis], 3, axis=1)

def R2E(R):
    sy = np.sqrt(R[0, 0, :] * R[0, 0, :] + R[1, 0, :] * R[1, 0, :])
    x = np.arctan2(R[2, 1, :], R[2, 2, :])
    y = np.arctan2(-R[2, 0, :], sy)
    z = np.arctan2(R[1, 0, :], R[0, 0, :])
    return np.array([x, y, z])
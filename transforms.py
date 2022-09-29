import numpy as np


# def get_optical_coords(M, feature):
#     """
#     fsu 0 cu 0
#     0 fsv cv 0
#     0 0 0 fsub
#     """
#     disparity = feature[0] - feature[2]
#     fsub = M[2, 3]
#     z = fsub / disparity
#     x = (feature[0] - M[0, 2]) * z / M[0, 0]
#     y = (feature[1] - M[1, 2]) * z / M[1, 1]
#     return np.array([[x, y, z,1]]).T
#
#
# def imuTcam(imu_T_cam, M, feature):
#     """
#
#     """
#     optical_coords = get_optical_coords(M, feature)
#     m = np.dot(imu_T_cam,optical_coords)
#     return m[:-1]

def get_optical_coords(M, feature):
    """
    fsu 0 cu 0
    0 fsv cv 0
    0 0 0 fsub
    """
    disparity = feature[0, :, :] - feature[2, :, :]  # 1xMxTs
    fsub = np.ones(disparity.shape) * M[2, 3]
    z = np.divide(fsub, disparity)  # 1xMxTs
    x = (feature[0, :, :] - M[0, 2]) * z / M[0, 0]
    y = (feature[1, :, :] - M[1, 2]) * z / M[1, 1]
    z = z.reshape(1, -1)
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    homogeneous = np.ones(x.shape)
    optical_coords = np.concatenate((x, y, z, homogeneous))
    return optical_coords


def imuTcam(imu_T_cam, M, feature):
    """
    features : 4 x M x Ts
    """
    optical_coords = get_optical_coords(M, feature)
    m = np.dot(imu_T_cam, optical_coords) #4 x (M*Ts)
    return m[:-1]

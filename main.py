import numpy as np
from pr3_utils import *
from tqdm import tqdm
from scipy import linalg


def get_optical_coords(M, feature):
    """
    fsu 0 cu 0
    0 fsv cv 0
    0 0 0 fsub
    """
    optical_coords = np.ones((4, feature.shape[1]))
    disparity = feature[0, :] - feature[2, :]  # 1xMxTs
    optical_coords[2, :] = M[2, 3] / disparity  # 1xMxTs
    optical_coords[0, :] = (feature[0, :] - M[0, 2]) * optical_coords[2, :] / M[0, 0]
    optical_coords[1, :] = (feature[1, :] - M[1, 2]) * optical_coords[2, :] / M[1, 1]
    return optical_coords


def get_world_coords(imu_T_cam, M, feature, world_T_imu):
    """
    features : 4 x M x Ts
    """
    optical_coords = get_optical_coords(M, feature)
    m = np.linalg.multi_dot([world_T_imu, imu_T_cam, optical_coords])  # 4 x (M*Ts)
    return m


def get_hat(x):
    """
    x is a 3x1 vector
    """
    x1, x2, x3 = x[0], x[1], x[2]
    x_hat = np.array([[0, -x3, x2],
                      [x3, 0, -x1],
                      [-x2, x1, 0]])
    return x_hat


def get_twist_hat(x):
    """
    x is a 6x1 vector
    """
    assert x.shape[0] == 6, "x is not a twist hat vector"
    pose = x[:3, 0].reshape(3, 1)
    theta_hat = get_hat(x[3:, 0])
    twist_hat = np.block([[theta_hat, pose], [np.zeros((1, 4))]])
    return twist_hat


def get_twist_adj(x):
    """
        x is a 6x1 vector
    """
    assert x.shape[0] == 6, "x is not a twist adj vector"
    v_hat = get_hat(x[:3, 0])
    w_hat = get_hat(x[3:, 0])
    return np.block([[w_hat, v_hat], [np.zeros((3, 3)), w_hat]])


def get_camera_parameters(K):
    fsu = K[0, 0]
    fsv = K[1, 1]
    cu = K[0, 2]
    cv = K[1, 2]
    return fsu, fsv, cu, cv


def get_projection(cam_T_world, mu_tj_homog):
    """
    Get projection of q
    """

    q = cam_T_world @ mu_tj_homog
    pi = q / q[2, :]
    return pi


def get_projection_dq(q):
    """
    Get dpi/dq
    """
    mat = np.eye(4)
    q = q.reshape(4, -1)
    mat[0, 2] = -q[0, :] / q[2, :]
    mat[1, 2] = -q[1, :] / q[2, :]
    mat[3, 2] = -q[3, :] / q[2, :]
    mat[2, 2] = 0
    return np.divide(mat, q[2, :])


def get_dot(s):
    """
    s : 4x1
    """
    row_1 = np.hstack([np.eye(3), -get_hat(s[:3])])
    dot_mat = np.vstack([row_1, np.zeros((1, 6))])
    return dot_mat


def get_joint_jacobian(Ks, n_features_ds, cam_T_world, cam_T_imu, imu_T_world, mu_tj_homog, features_tobe_updated, Nt):
    """
    Returns 4Ntx3M+6 joint jacobian
    It is calculated bu calculating observation Jacobian and pose Jacobian
    """
    Pt = np.vstack((np.eye(3), np.array([0, 0, 0])))
    assert Pt.shape == (4, 3)
    Ht = np.zeros((4 * Nt, 3 * n_features_ds + 6))
    for i in range(Nt):  # range of features to be updated
        j = features_tobe_updated[i]  # overall feature index value
        dq = get_projection_dq(cam_T_world @ mu_tj_homog[:, i])
        Hij_wrt_landmarks = np.linalg.multi_dot([Ks, dq, cam_T_world, Pt])
        Ht[4 * i:4 * (i + 1), 3 * j:3 * (j + 1)] = Hij_wrt_landmarks
        # imu_T_world is uj inv
        mu_mj = imu_T_world @ mu_tj_homog[:, i]
        mu_mt_dot = get_dot(mu_mj)
        dq = get_projection_dq(cam_T_imu @ mu_mj)
        Hij_wrt_pose = np.linalg.multi_dot([-Ks, dq, cam_T_imu, mu_mt_dot])  # 4x6
        Ht[4 * i:4 * (i + 1), -6:] = Hij_wrt_pose
    return Ht


def get_kalman_gain(Ht, joint_cov, Nt):
    """
    Return Kalman Gain
    """
    V = np.eye(4 * Nt) * 100
    Ht_transpose = Ht.T
    H_cov = joint_cov @ Ht_transpose
    inside_term = np.linalg.inv(Ht @ H_cov + V)
    return H_cov @ inside_term


def get_updated_cov(Ht, Kalman_Gain, joint_cov, n_features_ds, Nt):
    """
    Joseph form of covariance update
    """
    I = np.eye(3 * n_features_ds + 6)
    first_term = I - Kalman_Gain @ Ht
    V = np.eye(4 * Nt) * 100
    second_term = Kalman_Gain @ V @ Kalman_Gain.T
    return np.linalg.multi_dot([first_term, joint_cov, first_term.T]) + second_term


if __name__ == '__main__':
    #############################--HYPERPARAMETERS---######################################
    RUN_SLAM = False  # if False, load data
    DOWNSAMPLE = 5 # feature downsampling
    filename = "./data/10.npz"
    ##################################

    t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename)
    #imu_T_cam = imu_T_cam @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    cam_T_imu = np.linalg.inv(imu_T_cam)  # to convert IMU to camera frame
    n_features = features.shape[1]
    features = features[:, ::DOWNSAMPLE, :]
    n_features_ds = features.shape[1]
    t = t.T  # 3026x1

    n_ts = t.shape[0]

    # velocity - 3x3026
    l_var = np.std(linear_velocity, axis=1)
    a_var = np.std(angular_velocity, axis=1)
    motion_noise_cov = np.diag(np.array([l_var[0], l_var[1], l_var[2], a_var[0], a_var[1], a_var[2]]))
    fsu, fsv, cu, cv = get_camera_parameters(K)
    M = np.array([[fsu, 0, cu, 0],
                  [0, fsv, cv, 0],
                  [0, 0, 0, fsu * b]])
    Ks = np.array([[fsu, 0, cu, 0],
                   [0, fsv, cv, 0],
                   [fsu, 0, cu, -fsu * b],
                   [0, fsv, cv, 0]])
    # Initialize pose & landmarks
    world_T_imu = np.eye(4)  # 4x4
    imu_cov = 0.01 * np.eye(6)
    imu_trajectory = np.empty((4, 4, n_ts))  # store all pose matrices over time
    deadreckon = np.empty((4, 4, n_ts), dtype=np.float64)
    imu_trajectory[:, :, 0] = world_T_imu
    deadreckon[:, :, 0] = world_T_imu
    overall_landmarks_mu = np.zeros((3 * n_features_ds, 1))  # 3M
    overall_landmarks_cov = np.eye(3 * n_features_ds)  # cov init to identity
    observed_pixels = -1 * np.ones((4, n_features_ds))
    unobserved = np.array([-1, -1, -1, -1])
    joint_cov = np.block([[overall_landmarks_cov, np.zeros((3 * n_features_ds, 6)) * 1e-8],
                          [np.zeros((6, 3 * n_features_ds)) * 1e-8, imu_cov]])

    if RUN_SLAM:
        print("------------SLAM STARTED------------")
        for i in tqdm(range(1, n_ts)):
            tau = t[i] - t[i - 1]
            ut = np.vstack((linear_velocity[:, i].reshape(3, 1), angular_velocity[:, i].reshape(3, 1)))
            ut_hat = get_twist_hat(ut)  # 4x4
            ut_adj = get_twist_adj(ut)  # 6x6
            # (a) IMU Localization via EKF Prediction
            # Pose Kinematics
            world_T_imu = world_T_imu @ linalg.expm(tau * ut_hat)
            imu_T_world = np.linalg.inv(world_T_imu)
            deadreckon[:, :, i] = world_T_imu
            cam_T_world = cam_T_imu @ imu_T_world

            world_T_cam = np.linalg.inv(cam_T_world)
            motion_noise = np.random.multivariate_normal(np.array([0, 0, 0, 0, 0, 0]), motion_noise_cov).reshape(-1, 1)
            motion_noise_pert = linalg.expm(-tau * get_twist_adj(motion_noise))
            joint_cov[-6:, -6:] = np.linalg.multi_dot([motion_noise_pert, linalg.expm(-tau * ut_adj), \
                                                       joint_cov[-6:, -6:], linalg.expm(-tau * ut_adj).T,
                                                       motion_noise_pert.T])
            features_t = features[:, :, i]
            obs_feature_idx = tuple(np.where(np.sum(features_t, axis=0) != -4)[0])
            features_tobe_updated = list()

            if len(obs_feature_idx):
                obs_pixels_t = features_t[:, obs_feature_idx]
                m_world = get_world_coords(imu_T_cam, M, obs_pixels_t, world_T_imu)

                # check if landmark is encountered for the first time or have to update
                for m in range(len(obs_feature_idx)):
                    if np.array_equal(observed_pixels[:, obs_feature_idx[m]], unobserved):
                        observed_pixels[:, obs_feature_idx[m]] = np.array([-10, -1, -1, -1])
                        # this is done to change pixel coordinates of observed pixels
                        overall_landmarks_mu = overall_landmarks_mu.reshape(3, -1)
                        #remove homogenous coordinate
                        overall_landmarks_mu[:, obs_feature_idx[m]] = np.delete(m_world[:, m], 3, axis=0)

                    else:
                        features_tobe_updated.append(obs_feature_idx[m])
                Nt = len(features_tobe_updated)

                # (b) Landmark Mapping via EKF Update
                if Nt:
                    # mu_tj is mean of landmarks
                    mu_tj_homog = np.concatenate((overall_landmarks_mu[:, features_tobe_updated],
                                                  np.ones((1, len(features_tobe_updated)))), axis=0)
                    z_t = features_t[:, features_tobe_updated]
                    z_pred = np.dot(Ks, get_projection(cam_T_world, mu_tj_homog))
                    innovation = z_t - z_pred
                    innovation = innovation.reshape(-1, 1)
                    assert innovation.shape == (4 * Nt, 1)
                    # (c) Visual-Inertial SLAM
                    Ht = get_joint_jacobian(Ks, n_features_ds, cam_T_world, cam_T_imu, imu_T_world,
                                            mu_tj_homog, features_tobe_updated, Nt)
                    assert Ht.shape == (4 * Nt, 3 * n_features_ds + 6)

                    Kalman_Gain = get_kalman_gain(Ht, joint_cov, Nt)
                    assert Kalman_Gain.shape == (3 * n_features_ds + 6, 4 * Nt)
                    # NOTE I have combined part B and C in the code.
                    # The visual mapping is done by commenting code to update pose mean and covariance
                    # update pose and covariance of IMU
                    world_T_imu = (world_T_imu @ linalg.expm(get_twist_hat(Kalman_Gain[-6:, :] @ innovation)))
                    #joint_cov[-6:, -6:] = joint_cov[-6:, -6:] - Kalman_Gain[-6:, :] @ (Ht[:, -6:] @ joint_cov[-6:, -6:])
                    # update landmarks using landmark update eqn
                    overall_landmarks_mu = overall_landmarks_mu.reshape(-1, 1)
                    overall_landmarks_mu = overall_landmarks_mu + Kalman_Gain[:-6, :] @ innovation
                    overall_landmarks_mu = overall_landmarks_mu.reshape(3, -1)
                    #joint_cov[:-6, :-6] = joint_cov[:-6, :-6] - Kalman_Gain[:-6, :] @ (Ht[:, :-6] @ joint_cov[:-6, :-6])
                    # Uncomment to get joint covariance
                    # New formula using Joseph Form
                    #joint_cov = get_updated_cov(Ht, Kalman_Gain, joint_cov, n_features_ds, Nt)

                    # Formula from the slides. Uncomment to perform SLAM
                    joint_cov = (np.eye(3 * n_features_ds + 6) - Kalman_Gain @ Ht) @ joint_cov
            imu_trajectory[:, :, i] = world_T_imu
        #print(deadreckon)
        visualize_trajectory_2d(imu_trajectory, overall_landmarks_mu, show_landmarks=True, show_ori=True)

        np.save("imu_trajectory.npy", imu_trajectory)
        np.save("landmark_trajectory.npy", overall_landmarks_mu)
        print("------------SLAM COMPLETED------------")

    if not RUN_SLAM:
        with open('imu_trajectory_10.npy', 'rb') as poses:
            imu_trajectory = np.load(poses)
        with open('landmark_trajectory_10.npy', 'rb') as landmarks:
            overall_landmarks_mu = np.load(landmarks)
        visualize_trajectory_2d(imu_trajectory, overall_landmarks_mu, show_landmarks=True, show_ori=True)
        with open('imu_trajectory_3.npy', 'rb') as poses:
            imu_trajectory = np.load(poses)
        with open('landmark_trajectory_3.npy', 'rb') as landmarks:
            overall_landmarks_mu = np.load(landmarks)
        visualize_trajectory_2d(imu_trajectory, overall_landmarks_mu, show_landmarks=True, show_ori=True)

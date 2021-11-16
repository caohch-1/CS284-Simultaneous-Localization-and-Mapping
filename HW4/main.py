import numpy as np
from numpy import sin, cos
import time
from utils import rotate, project, fun, ba_matrix

# Intrinsic parameters
K = np.array([[500, 0, 320],
              [0, 500, 240],
              [0, 0, 1]])

# Camera Pose
theta_list = np.zeros((9, 3))
R_list = [np.identity(3) for i in range(9)]
C_list = [np.array([-10, -10, 0]),
          np.array([0, -10, 0]),
          np.array([10, -10, 0]),
          np.array([-10, 0, 0]),
          np.array([0, 0, 0]),
          np.array([10, 0, 0]),
          np.array([-10, 10, 0]),
          np.array([0, 10, 0]),
          np.array([10, 10, 0])]
T_list = [np.column_stack((np.identity(3), C.T)) for C in C_list]
P_list = [np.dot(K, np.dot(R_list[i], T_list[i])) for i in range(9)]
for i in range(9):
    P_list[i][-1][-1] = 1

# 2D measurement
uvList_list = [list(), list(), list(), list(), list(), list(), list(), list(), list()]
uv_list5 = [[50, 40, 1], [185, 40, 1], [320, 40, 1], [455, 40, 1], [590, 40, 1],
            [50, 140, 1], [185, 140, 1], [320, 140, 1], [455, 140, 1], [590, 140, 1],
            [50, 240, 1], [185, 240, 1], [320, 240, 1], [455, 240, 1], [590, 240, 1],
            [50, 340, 1], [185, 340, 1], [320, 340, 1], [455, 340, 1], [590, 340, 1],
            [50, 440, 1], [185, 440, 1], [320, 440, 1], [455, 440, 1], [590, 440, 1]]

# 3D measurement
xyz_list = [np.dot(np.linalg.pinv(K), np.array(uv).T) * 200 for uv in uv_list5]
xyz_list = [[xyz[0], xyz[1], xyz[2], 1] for xyz in xyz_list]


def task1(show_result=True):
    for i in range(9):
        for xyz in xyz_list:
            uvList_list[i].append(np.dot(P_list[i], xyz) / 200)
        for j in range(25):
            uvList_list[i][j][2] = 1

    if show_result:
        for i in range(9):
            print(i + 1)
            for u, v, _ in uvList_list[i]:
                print(u, end='\t')
            print()
            for u, v, _ in uvList_list[i]:
                print(v, end='\t')
            print()
    return uvList_list


# 2D measurement noise
def task21(show_results=True):
    for i in range(9):
        for uv in uvList_list[i]:
            uv[0] += np.random.normal(0, 0.25)
            uv[1] += np.random.normal(0, 0.25)

    if show_results:
        for i in range(9):
            print(i + 1)
            for u, v, _ in uvList_list[i]:
                print(u, end='\t')
            print()
            for u, v, _ in uvList_list[i]:
                print(v, end='\t')
            print()
    return uvList_list


# 3D measurement noise
def task22(show_results=True):
    for xyz in xyz_list:
        xyz[0] += np.random.normal(0, 2)
        xyz[1] += np.random.normal(0, 2)
        xyz[2] += np.random.normal(0, 2)

    if show_results:
        for xyz in xyz_list:
            print(xyz, end='\t')
        print()
    return xyz_list


# Camera Pose noise
def task23(show_results=True):
    theta_list = list()
    for C in C_list:
        C[0] += np.random.normal(0, 2)
        C[1] += np.random.normal(0, 2)

    for i in range(9):
        alpha = np.random.normal(0, 0.002)
        beta = np.random.normal(0, 0.002)
        gamma = np.random.normal(0, 0.002)
        theta_list.append([alpha, beta, gamma])
        R_list[i] = np.dot(np.array([[cos(alpha) * cos(gamma) - cos(beta) * sin(alpha) * sin(gamma),
                                      -cos(alpha) * sin(gamma) - cos(beta) * sin(alpha) * cos(gamma),
                                      sin(beta) * sin(alpha)],
                                     [sin(alpha) * cos(gamma) + cos(beta) * cos(alpha) * sin(gamma),
                                      -sin(alpha) * sin(gamma) + cos(beta) * cos(alpha) * cos(gamma),
                                      -sin(beta) * cos(alpha)],
                                     [sin(beta) * sin(gamma), sin(beta) * cos(gamma), cos(beta)]]), R_list[i])

    if show_results:
        for R in R_list:
            print(R)

        for C in C_list:
            print(C)
    return R_list, C_list, theta_list


if __name__ == '__main__':
    t0 = time.time()
    uvList_list = task1(False)
    uvList_list = task21(False)
    xyz_list = task22(False)
    R_list, C_list, theta_list = task23(False)

    camera_params = np.zeros((9, 9))
    for i in range(9):
        camera_params[i] = [theta_list[i][0], theta_list[i][0], theta_list[i][0], C_list[i][0], C_list[i][1],
                            C_list[i][2], K[0][0], 0, 0]

    points_3d = np.zeros((25, 3))
    for i in range(25):
        points_3d[i] = xyz_list[i][:-1]

    camera_indices = np.ndarray(shape=(9 * 25,), dtype=np.int)
    point_indices = np.ndarray(shape=(9 * 25,), dtype=np.int)
    from scipy.optimize import least_squares
    for i in range(9):
        for j in range(25):
            camera_indices[i * 25 + j] = i
            point_indices[i * 25 + j] = j

    points_2d = np.zeros((9 * 25, 2))
    for i in range(9):
        for j in range(25):
            points_2d[i * 25 + j] = [uvList_list[i][j][0], uvList_list[i][j][1]]

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = 9 * n_cameras + 3 * n_points

    m = 2 * points_2d.shape[0]

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    n *= 10
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

    A = ba_matrix(n_cameras, n_points, camera_indices, point_indices)
    res = least_squares(fun, x0, jac_sparsity=A, args=(n_cameras, n_points, camera_indices, point_indices, points_2d))

    print('Time consumed {}s'.format(time.time()-t0))
    print('3D points err: ', np.abs(x0[9 * n_cameras:] - res.x[9 * n_cameras:]).sum() / n)
    pose = (x0[:9 * n_cameras] - res.x[:9 * n_cameras]).reshape((9, 9))
    print('Pose Rotation err: ', np.abs(pose[:, 0:3]).sum() / (3*9))
    print('Pose Translation err: ', np.abs(pose[:, 3:6]).sum() / (3*9*6))

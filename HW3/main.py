import numpy as np
from matplotlib import pyplot as plt
import os
import scipy.optimize
import cv2

np.set_printoptions(precision=4)


def CPM_DE(projection_matrix):
    P = projection_matrix

    M = P[:, 0:3]
    K, R = scipy.linalg.rq(M)

    T = np.diag(np.sign(np.diag(K)))
    if scipy.linalg.det(T) < 0:
        T[1, 1] *= -1

    K = np.dot(K, T)
    R = np.dot(R, T)

    C = np.dot(scipy.linalg.inv(-M), P[:, 3])

    return K, R, C


def find_cs(M, R):
    c = - M[2][2] / np.sqrt(pow(M[2][1], 2) + pow(M[2][2], 2))
    s = M[2][1] / np.sqrt(pow(M[2][1], 2) + pow(M[2][2], 2))
    c_ = R[2][2] / c
    s_ = R[2][0]
    c__ = R[0][0] / c_
    s__ = R[1][0] / (-c_)

    Rx = np.array([[1, 0, 0],
                   [0, c, -s],
                   [0, s, c]])
    Ry = np.array([[c_, 0, s_],
                   [0, 1, 0],
                   [-s_, 0, c_]])
    Rz = np.array([[c__, -s__, 0],
                   [s__, c__, 0],
                   [0, 0, 1]])
    return [c, c_, c__, s, s_, s__, Rx, Ry, Rz]


def Normalization(nd, x):
    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)
    if nd == 2:
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])

    Tr = np.linalg.inv(Tr)
    x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x[0:nd, :].T

    return Tr, x


def DLT(nd, xyz, uv):
    Txyz, xyzn = Normalization(nd, xyz)
    Tuv, uvn = Normalization(2, uv)

    A = []
    if nd == 2:  # 2D DLT
        for i in range(xyz.shape[0]):
            x, y = xyzn[i, 0], xyzn[i, 1]
            u, v = uvn[i, 0], uvn[i, 1]
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    elif nd == 3:  # 3D DLT
        for i in range(xyz.shape[0]):
            x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
            u, v = uvn[i, 0], uvn[i, 1]
            A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
            A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])

    A = np.asarray(A)

    U, S, Vh = np.linalg.svd(A)

    L = Vh[-1, :] / Vh[-1, -1]

    H = L.reshape(3, nd + 1)

    H = np.dot(np.dot(np.linalg.pinv(Tuv), H), Txyz);
    H = H / H[-1, -1]
    L = H.reshape((3, 4))

    uv2 = np.dot(H, np.concatenate((xyz.T, np.ones((1, xyz.shape[0])))))
    uv2 = uv2 / uv2[2, :]

    err = np.sqrt(np.mean(np.sum((uv2[0:2, :].T - uv) ** 2, 1)))

    return L, err


if __name__ == '__main__':
    # Task1
    print('=' * 15 + 'Task1' + '=' * 15)
    P = np.array([[3.53553e+2, 3.39645e+2, 2.77744e+2, -1.44946e+6],
                  [-1.03528e+2, 2.33212e+1, 4.59607e+2, -6.3252e+5],
                  [7.07107e-1, -3.53553e-1, 6.12372e-1, -9.18559e+2]])
    M = P[:, 0:3]
    K, R, C = CPM_DE(P)
    cs_list = find_cs(M, R)
    print('Intrinsics Matrix:')
    print(K)
    print('\nRotation Matrix:')
    print(R)
    print('\nC:')
    print(C)

    # Task2
    print('\n\n'+'='*15+'Task2'+'='*15)
    xyz = np.array([[0, 0, 104.2], [180, 0, 104.2], [180, -90, 104.2], [0, 0, 101.5], [0, -1, 0], [175, 0, 0]])

    img1 = cv2.imread('./CS284_hw3_data/img1.jpeg')
    img1 = cv2.circle(img1, (885, 830), 20, (255, 0, 0), 4)  # 0, 0, 104.2
    img1 = cv2.circle(img1, (3628, 340), 20, (255, 165, 0), 4)  # 180, 0, 104.2
    img1 = cv2.circle(img1, (2626, 240), 20, (139, 129, 76), 4)  # 180, -90, 104.2
    img1 = cv2.circle(img1, (896, 895), 20, (0, 255, 0), 4)  # 0, 0, 101.5
    img1 = cv2.circle(img1, (1273, 2809), 20, (0, 0, 255), 4)  # 0, -1, 0
    img1 = cv2.circle(img1, (3286, 1715), 20, (160, 32, 240), 4)  # 175, 0, 0
    cv2.imwrite('./img1_corner.jpeg', img1)
    uv1 = [[885, 830], [3628, 340], [2626, 240], [896, 895], [1273, 2809], [3286, 1715]]

    img2 = cv2.imread('./CS284_hw3_data/img2.jpeg')
    img2 = cv2.circle(img2, (47, 1037), 20, (255, 0, 0), 4)  # 0, 0, 104.2
    img2 = cv2.circle(img2, (3795, 980), 20, (255, 165, 0), 4)  # 180, 0, 104.2
    img2 = cv2.circle(img2, (3097, 695), 20, (139, 129, 76), 4)  # 180, -90, 104.2
    img2 = cv2.circle(img2, (56, 1087), 20, (0, 255, 0), 4)  # 0, 0, 101.5
    img2 = cv2.circle(img2, (505, 2727), 20, (0, 0, 255), 4)  # 0, -1, 0
    img2 = cv2.circle(img2, (3333, 2588), 20, (160, 32, 240), 4)  # 175, 0, 0
    cv2.imwrite('./img2_corner.jpeg', img2)
    uv2 = [[47, 1037], [3795, 980], [3097, 695], [56, 1087], [505, 2727], [3333, 2588]]

    img3 = cv2.imread('./CS284_hw3_data/img3.jpeg')
    img3 = cv2.circle(img3, (687, 819), 20, (255, 0, 0), 4)  # 0, 0, 104.2
    img3 = cv2.circle(img3, (2735, 1175), 20, (255, 165, 0), 4)  # 180, 0, 104.2
    img3 = cv2.circle(img3, (3275, 896), 20, (139, 129, 76), 4)  # 180, -90, 104.2
    img3 = cv2.circle(img3, (687, 849), 20, (0, 255, 0), 4)  # 0, 0, 101.5
    img3 = cv2.circle(img3, (877, 1929), 20, (0, 0, 255), 4)  # 0, -1, 0
    img3 = cv2.circle(img3, (2527, 2760), 20, (160, 32, 240), 4)  # 175, 0, 0
    cv2.imwrite('./img3_corner.jpeg', img3)
    uv3 = [[687, 819], [2735, 1175], [3275, 896], [687, 849], [877, 1929], [2527, 2760]]

    L1, err1 = DLT(3, xyz, uv1)
    K1, _, _ = CPM_DE(L1)
    print('Intrinsics Matrix For Img1: ')
    print(K1)
    print('Error For Img1: ', err1)

    L2, err2 = DLT(3, xyz, uv2)
    K2, _, _ = CPM_DE(L2)
    print('\nIntrinsics Matrix For Img2: ')
    print(K2)
    print('Error For Img2: ', err2)

    L3, err3 = DLT(3, xyz, uv3)
    K3, _, _ = CPM_DE(L3)
    print('\nIntrinsics Matrix For Img3: ')
    print(K3)
    print('Error For Img3: ', err3)

    print('\nError between K1 and K2: ', np.abs(K1-K2)[:, :2].sum())
    print('Error between K2 and K3: ', np.abs(K2-K3)[:, :2].sum())
    print('Error between K3 and K1: ', np.abs(K3-K1)[:, :2].sum())

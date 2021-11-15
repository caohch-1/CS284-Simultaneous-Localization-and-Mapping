import numpy as np
from matplotlib import pyplot as plt
import os
import scipy.optimize
import cv2
import time


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


def calAndDeEssen(pts1, pts2, K1, K2):
    f_avg = (K1[0, 0] + K2[0, 0]) / 2
    pts1, pts2 = np.ascontiguousarray(pts1, np.float32), np.ascontiguousarray(pts2, np.float32)

    pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K1, distCoeffs=None)
    pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K2, distCoeffs=None)

    E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.),
                                   method=cv2.RANSAC, prob=0.999, threshold=3.0 / f_avg)
    points, R, t, mask_pose = cv2.recoverPose(E, pts_l_norm, pts_r_norm)
    return E, R, t


if __name__ == '__main__':
    # Task1
    st = time.time()
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
    print('\nConsumed Time: {}'.format(time.time()-st))

    # Task2
    st = time.time()
    print('\n\n' + '=' * 15 + 'Task2' + '=' * 15)
    xyz = np.array([[0, 0, 104.2], [180, 0, 104.2], [180, -90, 104.2], [0, 0, 101.5], [0, -1, 0], [175, 0, 0]])

    img1 = cv2.imread('./CS284_hw3_data/img1.jpeg')
    img1 = cv2.circle(img1, (885, 830), 20, (255, 0, 0), 4)  # 0, 0, 104.2
    img1 = cv2.circle(img1, (3628, 340), 20, (255, 165, 0), 4)  # 180, 0, 104.2
    img1 = cv2.circle(img1, (2626, 240), 20, (139, 129, 76), 4)  # 180, -90, 104.2
    img1 = cv2.circle(img1, (896, 895), 20, (0, 255, 0), 4)  # 0, 0, 101.5
    img1 = cv2.circle(img1, (1273, 2809), 20, (0, 0, 255), 4)  # 0, -1, 0
    img1 = cv2.circle(img1, (3286, 1715), 20, (160, 32, 240), 4)  # 175, 0, 0
    cv2.imwrite('./vis/img1_corner.jpeg', img1)
    uv1 = [[885, 830], [3628, 340], [2626, 240], [896, 895], [1273, 2809], [3286, 1715]]

    img2 = cv2.imread('./CS284_hw3_data/img2.jpeg')
    img2 = cv2.circle(img2, (47, 1037), 20, (255, 0, 0), 4)  # 0, 0, 104.2
    img2 = cv2.circle(img2, (3795, 980), 20, (255, 165, 0), 4)  # 180, 0, 104.2
    img2 = cv2.circle(img2, (3097, 695), 20, (139, 129, 76), 4)  # 180, -90, 104.2
    img2 = cv2.circle(img2, (56, 1087), 20, (0, 255, 0), 4)  # 0, 0, 101.5
    img2 = cv2.circle(img2, (505, 2727), 20, (0, 0, 255), 4)  # 0, -1, 0
    img2 = cv2.circle(img2, (3333, 2588), 20, (160, 32, 240), 4)  # 175, 0, 0
    cv2.imwrite('./vis/img2_corner.jpeg', img2)
    uv2 = [[47, 1037], [3795, 980], [3097, 695], [56, 1087], [505, 2727], [3333, 2588]]

    img3 = cv2.imread('./CS284_hw3_data/img3.jpeg')
    img3 = cv2.circle(img3, (687, 819), 20, (255, 0, 0), 4)  # 0, 0, 104.2
    img3 = cv2.circle(img3, (2735, 1175), 20, (255, 165, 0), 4)  # 180, 0, 104.2
    img3 = cv2.circle(img3, (3275, 896), 20, (139, 129, 76), 4)  # 180, -90, 104.2
    img3 = cv2.circle(img3, (687, 849), 20, (0, 255, 0), 4)  # 0, 0, 101.5
    img3 = cv2.circle(img3, (877, 1929), 20, (0, 0, 255), 4)  # 0, -1, 0
    img3 = cv2.circle(img3, (2527, 2760), 20, (160, 32, 240), 4)  # 175, 0, 0
    cv2.imwrite('./vis/img3_corner.jpeg', img3)
    uv3 = [[687, 819], [2735, 1175], [3275, 896], [687, 849], [877, 1929], [2527, 2760]]

    L1, err1 = DLT(3, xyz, uv1)
    K1, R1, C1 = CPM_DE(L1)
    print('Intrinsics Matrix For Img1: ')
    print(K1)
    print('Error For Img1: ', err1)

    L2, err2 = DLT(3, xyz, uv2)
    K2, R2, C2 = CPM_DE(L2)
    print('\nIntrinsics Matrix For Img2: ')
    print(K2)
    print('Error For Img2: ', err2)

    L3, err3 = DLT(3, xyz, uv3)
    K3, R3, C3 = CPM_DE(L3)
    print('\nIntrinsics Matrix For Img3: ')
    print(K3)
    print('Error For Img3: ', err3)

    print('\nError between K1 and K2: ', np.abs(K1 - K2)[:, :2].sum())
    print('Error between K2 and K3: ', np.abs(K2 - K3)[:, :2].sum())
    print('Error between K3 and K1: ', np.abs(K3 - K1)[:, :2].sum())
    print('\nConsumed Time: {}'.format(time.time() - st))

    # Task3
    st = time.time()
    print('\n\n' + '=' * 15 + 'Task3' + '=' * 15)
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    img1 = cv2.imread('./CS284_hw3_data/img1.jpeg', 0)
    img2 = cv2.imread('./CS284_hw3_data/img2.jpeg', 0)
    img3 = cv2.imread('./CS284_hw3_data/img3.jpeg', 0)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    kp3, des3 = orb.detectAndCompute(img2, None)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    min_match = matches[0].distance
    matches = [match for match in matches if match.distance <= max(2 * min_match, 30.0)]
    img_match12 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    cv2.imwrite('./vis/match_img12.jpeg', img_match12)

    matches = bf.match(des2, des1)
    matches = sorted(matches, key=lambda x: x.distance)
    min_match = matches[0].distance
    matches = [match for match in matches if match.distance <= max(2 * min_match, 30.0)]
    img_match21 = cv2.drawMatches(img2, kp2, img1, kp1, matches, None, flags=2)
    cv2.imwrite('./vis/match_img21.jpeg', img_match21)

    matches = bf.match(des1, des3)
    matches = sorted(matches, key=lambda x: x.distance)
    min_match = matches[0].distance
    matches = [match for match in matches if match.distance <= max(2 * min_match, 30.0)]
    img_match13 = cv2.drawMatches(img1, kp1, img3, kp3, matches, None, flags=2)
    cv2.imwrite('./vis/match_img13.jpeg', img_match13)

    matches = bf.match(des3, des1)
    matches = sorted(matches, key=lambda x: x.distance)
    min_match = matches[0].distance
    matches = [match for match in matches if match.distance <= max(2 * min_match, 30.0)]
    img_match31 = cv2.drawMatches(img3, kp3, img1, kp1, matches, None, flags=2)
    cv2.imwrite('./vis/match_img31.jpeg', img_match31)

    matches = bf.match(des2, des3)
    matches = sorted(matches, key=lambda x: x.distance)
    min_match = matches[0].distance
    matches = [match for match in matches if match.distance <= max(2 * min_match, 30.0)]
    img_match23 = cv2.drawMatches(img2, kp2, img3, kp3, matches, None, flags=2)
    cv2.imwrite('./vis/match_img23.jpeg', img_match23)

    matches = bf.match(des3, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    min_match = matches[0].distance
    matches = [match for match in matches if match.distance <= max(2 * min_match, 30.0)]
    img_match32 = cv2.drawMatches(img3, kp3, img2, kp2, matches, None, flags=2)
    cv2.imwrite('./vis/match_img32.jpeg', img_match32)

    match12_p1s = list()
    match12_p2s = list()
    for i in range(len(matches)):
        match12_p1s.append(kp1[matches[i].queryIdx].pt)
        match12_p2s.append(kp2[matches[i].queryIdx].pt)
    E12, R12, t12 = calAndDeEssen(match12_p1s, match12_p2s, K1, K2)

    match13_p1s = list()
    match13_p2s = list()
    for i in range(len(matches)):
        match13_p1s.append(kp1[matches[i].queryIdx].pt)
        match13_p2s.append(kp2[matches[i].queryIdx].pt)
    E13, R13, t13 = calAndDeEssen(match13_p1s, match13_p2s, K1, K3)

    match23_p1s = list()
    match23_p2s = list()
    for i in range(len(matches)):
        match23_p1s.append(kp1[matches[i].queryIdx].pt)
        match23_p2s.append(kp2[matches[i].queryIdx].pt)
    E23, R23, t23 = calAndDeEssen(match23_p1s, match23_p2s, K2, K3)

    match21_p1s = list()
    match21_p2s = list()
    for i in range(len(matches)):
        match21_p1s.append(kp1[matches[i].queryIdx].pt)
        match21_p2s.append(kp2[matches[i].queryIdx].pt)
    E21, R21, t21 = calAndDeEssen(match21_p1s, match21_p2s, K2, K1)

    match31_p1s = list()
    match31_p2s = list()
    for i in range(len(matches)):
        match31_p1s.append(kp1[matches[i].queryIdx].pt)
        match31_p2s.append(kp2[matches[i].queryIdx].pt)
    E31, R31, t31 = calAndDeEssen(match31_p1s, match31_p2s, K3, K1)

    match32_p1s = list()
    match32_p2s = list()
    for i in range(len(matches)):
        match32_p1s.append(kp1[matches[i].queryIdx].pt)
        match32_p2s.append(kp2[matches[i].queryIdx].pt)
    E32, R32, t32 = calAndDeEssen(match32_p1s, match32_p2s, K3, K2)

    A1 = np.zeros((3, 4))
    A1[:, 0:3] = R1
    A1[:, 3:4] = C1.reshape((3, 1))

    A2 = np.zeros((3, 4))
    A2[:, 0:3] = R2
    A2[:, 3:4] = C2.reshape((3, 1))

    A3 = np.zeros((3, 4))
    A3[:, 0:3] = R3
    A3[:, 3:4] = C3.reshape((3, 1))

    print('Transform Matrix Error for Img1-Img2: ', np.abs(np.dot(A2, np.linalg.pinv(A1)) - R12).sum())
    print('Transform Matrix Error for Img1-Img3: ', np.abs(np.dot(A3, np.linalg.pinv(A1)) - R13).sum())
    print('Transform Matrix Error for Img2-Img3: ', np.abs(np.dot(A3, np.linalg.pinv(A2)) - R23).sum())
    print('Transform Matrix Error for Img2-Img1: ', np.abs(np.dot(A1, np.linalg.pinv(A2)) - R21).sum())
    print('Transform Matrix Error for Img3-Img1: ', np.abs(np.dot(A1, np.linalg.pinv(A3)) - R31).sum())
    print('Transform Matrix Error for Img3-Img2: ', np.abs(np.dot(A2, np.linalg.pinv(A3)) - R32).sum())
    print('\nConsumed Time: {}'.format(time.time() - st))


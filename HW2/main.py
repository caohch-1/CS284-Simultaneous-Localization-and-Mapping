import numpy as np
import math
from matplotlib import pyplot as plt
import sys


def read_data(path='./CS284_hw2_data/T_adjacent.txt'):
    with open(path) as f:
        raw = f.readlines()
        raw = [row.split('\t') for row in raw]
        raw = [[float(raw[j][i]) for j in range(3)] for i in range(len(raw[0]) - 1)]
        return raw


def xytheta2T(xytheta: list):
    x, y, theta = xytheta
    T = [[math.cos(theta), -math.sin(theta), x],
         [math.sin(theta), math.cos(theta), y],
         [0, 0, 1]]
    return T


def g2o2xytheta(file_path="task2_g2o.g2o", point_index=0):
    with open(file_path, 'r') as f:
        data = f.readlines()[point_index]
        data = data.strip().split(' ')
        return [float(d) for d in data[-3:]]


def calT(previous_point: list, current_point: list):
    T = np.dot(current_point, np.linalg.pinv(previous_point))
    return T


def g2o_optimize(path="task2_g2o.g2o", out="res_task2.g2o"):
    from graphslam.load import load_g2o_se2

    g = load_g2o_se2(path)
    g.plot()
    g.calc_chi2()
    g.optimize()
    g.plot()
    g.to_g2o(out)


def task1():
    xytheta_list = read_data()
    T_list = [xytheta2T(xytheta) for xytheta in xytheta_list]
    frame_list = [np.identity(3)]
    for i in range(len(T_list)):
        frame_list.append(np.dot(T_list[i], frame_list[-1]))

    for i in range(len(frame_list)):
        print('-' * 18 + str(i + 1) + '-' * 18)
        print(frame_list[i])


def task2():
    xytheta_list = read_data()
    T_list = [xytheta2T(xytheta) for xytheta in xytheta_list][:-1]
    frame_list = [np.identity(3)]
    for i in range(len(T_list)):
        frame_list.append(np.dot(T_list[i], frame_list[-1]))

    vertex_info_list = list()

    for i in range(len(frame_list))[:6]:
        x = frame_list[i][0][2]
        y = frame_list[i][1][2]
        theta = math.acos(frame_list[i][0][0])
        vertex_info_list.append('VERTEX_SE2 {} {:.6f} {:.6f} {:.6f}\n'.format(i, x, y, theta))

    for i in range(len(frame_list))[6:]:
        x = frame_list[i][0][2]
        y = frame_list[i][1][2]
        theta = -math.acos(frame_list[i][0][0])
        vertex_info_list.append('VERTEX_SE2 {} {:.6f} {:.6f} {:.6f}\n'.format(i, x, y, theta))

    edge_info_list = list()

    for i in range(len(xytheta_list) - 1):
        x = xytheta_list[i][0]
        y = xytheta_list[i][1]
        theta = xytheta_list[i][2]
        edge_info_list.append(
            'EDGE_SE2 {} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(i, (i + 1) % 12, x,
                                                                                                     y, theta, 1, 0, 0,
                                                                                                     1, 0, 1))

    with open('task2_g2o.g2o', 'w') as f:
        f.writelines(vertex_info_list)
        f.writelines(edge_info_list)

    g2o_optimize()


def task3():
    xytheta_list = read_data()
    T_list = [xytheta2T(xytheta) for xytheta in xytheta_list][:-1]
    frame_list = [np.identity(3)]
    for i in range(len(T_list)):
        frame_list.append(np.dot(T_list[i], frame_list[-1]))

    vertex_info_list = list()

    for i in range(len(frame_list))[:6]:
        x = frame_list[i][0][2]
        y = frame_list[i][1][2]
        theta = math.acos(frame_list[i][0][0])
        vertex_info_list.append('VERTEX_SE2 {} {:.6f} {:.6f} {:.6f}\n'.format(i, x, y, theta))

    for i in range(len(frame_list))[6:]:
        x = frame_list[i][0][2]
        y = frame_list[i][1][2]
        theta = -math.acos(frame_list[i][0][0])
        vertex_info_list.append('VERTEX_SE2 {} {:.6f} {:.6f} {:.6f}\n'.format(i, x, y, theta))

    edge_info_list1 = list()

    for i in range(len(xytheta_list) - 1):
        x = xytheta_list[i][0]
        y = xytheta_list[i][1]
        theta = xytheta_list[i][2]
        edge_info_list1.append(
            'EDGE_SE2 {} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(i, (i + 1) % 12, x,
                                                                                                     y, theta, 1, 0, 0,
                                                                                                     1, 0, 1))

    edge_info_list2 = list()
    xytheta_list = read_data('./CS284_hw2_data/T_everytwo.txt')
    for i in range(len(xytheta_list)):
        x = xytheta_list[i][0]
        y = xytheta_list[i][1]
        theta = xytheta_list[i][2]
        edge_info_list1.append(
            'EDGE_SE2 {} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(i, (i + 2) % 12, x,
                                                                                                     y, theta, 1, 0, 0,
                                                                                                     1, 0, 1))

    with open('task3_g2o.g2o', 'w') as f:
        f.writelines(vertex_info_list)
        f.writelines(edge_info_list1)
        f.writelines(edge_info_list2)

    g2o_optimize('task3_g2o.g2o', 'res_task3.g2o')


def test():
    g2o_optimize('input_MITb_g2o.g2o', 'output_MITb_g2o.g2o')


if __name__ == '__main__':
    # p0 = xytheta2T(g2o2xytheta(file_path='task2_g2o.g2o', point_index=10))
    # p1 = xytheta2T(g2o2xytheta(file_path='task2_g2o.g2o', point_index=11))
    # print('-' * 18 + 'Unoptimized' + '-' * 18)
    # print(calT(p0, p1).tolist())
    #
    # p0 = xytheta2T(g2o2xytheta(file_path='res_task2.g2o', point_index=10))
    # p1 = xytheta2T(g2o2xytheta(file_path='res_task2.g2o', point_index=11))
    # print('-' * 18 + 'Optimized' + '-' * 18)
    # print(calT(p0, p1).tolist())
    #
    # print('-' * 18 + 'Ref' + '-' * 18)
    # print(xytheta2T(read_data()[11]))
    test()
    task1()
    task2()
    task3()

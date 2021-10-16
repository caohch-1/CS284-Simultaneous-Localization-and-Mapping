import numpy as np
import math
from matplotlib import pyplot as plt
import sys


def read_data(path='./CS284_hw2_data/T_adjacent.txt'):
    with open(path) as f:
        raw = f.readlines()
        raw = [row.split('\t') for row in raw]
        raw = [[float(raw[j][i]) for j in range(3)] for i in range(12)]
        return raw


def xytheta2T(xytheta: list):
    x, y, theta = xytheta
    T = [[math.cos(theta), -math.sin(theta), x],
         [math.sin(theta), math.cos(theta), y],
         [0, 0, 1]]
    return T


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

    with open('task2_g2o.g2o', 'w') as f:
        f.writelines(vertex_info_list)

    edge_info_list = list()


if __name__ == '__main__':
    task1()
    task2()

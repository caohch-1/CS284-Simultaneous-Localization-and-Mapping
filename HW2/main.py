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


if __name__ == '__main__':
    task2()

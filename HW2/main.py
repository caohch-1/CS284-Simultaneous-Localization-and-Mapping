import os.path

import numpy as np
import math
import argparse
import time

parser = argparse.ArgumentParser(description='Choose task')
parser.add_argument('--task_index', type=str, default='3',
                    help='0 for MIT test task\n1 for task1\n2 for task2\n3 for task3')

ref1 = [[0, 0, math.pi * 0 / 6],
        [0, 0, math.pi * 1 / 6],
        [0, 0, math.pi * 2 / 6],
        [0, 0, math.pi * 3 / 6],
        [0, 0, math.pi * 4 / 6],
        [0, 0, math.pi * 5 / 6],
        [0, 0, -math.pi * 6 / 6],
        [0, 0, -math.pi * 5 / 6],
        [0, 0, -math.pi * 4 / 6],
        [0, 0, -math.pi * 3 / 6],
        [0, 0, -math.pi * 2 / 6],
        [0, 0, -math.pi * 1 / 6]]

ref2 = [[0, 0, math.pi * 0 / 6],
        [0, 0, math.pi * 1 / 6],
        [0, 0, math.pi * 2 / 6],
        [0, 0, math.pi * 3 / 6],
        [0, 0, math.pi * 4 / 6],
        [0, 0, math.pi * 5 / 6],
        [0, 0, math.pi * 6 / 6],
        [0, 0, -math.pi * 5 / 6],
        [0, 0, -math.pi * 4 / 6],
        [0, 0, -math.pi * 3 / 6],
        [0, 0, -math.pi * 2 / 6],
        [0, 0, -math.pi * 1 / 6]]

ref3 = [[0, 0, math.pi * 0 / 6],
        [0, 0, math.pi * 1 / 6],
        [0, 0, math.pi * 2 / 6],
        [0, 0, math.pi * 3 / 6],
        [0, 0, math.pi * 4 / 6],
        [0, 0, math.pi * 5 / 6],
        [0, 0, math.pi * 6 / 6],
        [0, 0, -math.pi * 5 / 6],
        [0, 0, -math.pi * 4 / 6],
        [0, 0, -math.pi * 3 / 6],
        [0, 0, -math.pi * 2 / 6],
        [0, 0, -math.pi * 1 / 6],
        [0, 0, math.pi * 0 / 6]]

ref4 = [[0, 0, math.pi * 0 / 6],
        [0, 0, math.pi * 1 / 6],
        [0, 0, math.pi * 2 / 6],
        [0, 0, math.pi * 3 / 6],
        [0, 0, math.pi * 4 / 6],
        [0, 0, math.pi * 5 / 6],
        [0, 0, math.pi * 6 / 6],
        [0, 0, -math.pi * 5 / 6],
        [0, 0, -math.pi * 4 / 6],
        [0, 0, -math.pi * 3 / 6],
        [0, 0, -math.pi * 2 / 6],
        [0, 0, -math.pi * 1 / 6],
        [0, 0, math.pi * 0 / 6]]


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


def g2o2xytheta(file_path="g2o_data/task2_g2o.g2o", point_index=0):
    with open(file_path, 'r') as f:
        data = f.readlines()[point_index]
        data = data.strip().split(' ')
        return [float(d) for d in data[-3:]]


def calT(previous_point: list, current_point: list):
    T = np.dot(current_point, np.linalg.pinv(previous_point))
    return T


def g2o_optimize(path="g2o_data/task2_g2o.g2o", out="g2o_data/res_task2.g2o"):
    from g2opy.load import load_g2o_se2
    g = load_g2o_se2(path)
    g.plot(title='Unoptimized')
    g.calc_chi2()
    g.optimize()
    g.plot(title='Optimized')
    g.to_g2o(out)


def g2o_draw(path="g2o_data/task2_g2o.g2o"):
    from g2opy.load import load_g2o_se2
    g = load_g2o_se2(path)
    g.plot(title='Unoptimized')


def task1():
    start_time = time.time()

    xytheta_list = read_data()
    T_list = [xytheta2T(xytheta) for xytheta in xytheta_list]
    frame_list = [np.identity(3)]
    for i in range(len(T_list)):
        frame_list.append(np.dot(T_list[i], frame_list[-1]))

    for i in range(len(frame_list)):
        print('-' * 18 + str(i + 1) + '-' * 18)
        print(frame_list[i])

    if not os.path.exists('g2o_data/task1_g2o.g2o'):
        xytheta_list = read_data()
        T_list = [xytheta2T(xytheta) for xytheta in xytheta_list]
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

        for i in range(len(xytheta_list)):
            x = xytheta_list[i][0]
            y = xytheta_list[i][1]
            theta = xytheta_list[i][2]
            edge_info_list.append(
                'EDGE_SE2 {} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(i,
                                                                                                         (i + 1) % 13,
                                                                                                         x,
                                                                                                         y, theta, 1, 0,
                                                                                                         0,
                                                                                                         1, 0, 1))

        with open('g2o_data/task1_g2o.g2o', 'w') as f:
            f.writelines(vertex_info_list)
            f.writelines(edge_info_list)
        g2o_draw('g2o_data/task1_g2o.g2o')
    end_time = time.time()
    print('Time consumed: {}'.format(end_time - start_time))


def task2():
    start_time = time.time()

    if not os.path.exists('g2o_data/task2_g2o.g2o'):
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
                'EDGE_SE2 {} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(i,
                                                                                                         (i + 1) % 12,
                                                                                                         x,
                                                                                                         y, theta, 1, 0,
                                                                                                         0,
                                                                                                         1, 0, 1))

        with open('g2o_data/task2_g2o.g2o', 'w') as f:
            f.writelines(vertex_info_list)
            f.writelines(edge_info_list)

    g2o_optimize()

    end_time = time.time()
    print('Time consumed: {}'.format(end_time - start_time))
    T12_1 = calT(xytheta2T(g2o2xytheta(point_index=11)), xytheta2T(g2o2xytheta(point_index=0)))

    print('T12,1: ', T12_1[0][2], T12_1[1][2], math.acos(T12_1[0][0]), '\n')


def task3():
    start_time = time.time()

    if not os.path.exists('g2o_data/task3_g2o.g2o'):
        xytheta_list = read_data()
        T_list = [xytheta2T(xytheta) for xytheta in xytheta_list]
        frame_list = [np.identity(3)]
        for i in range(len(T_list) - 1):
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

        for i in range(len(xytheta_list)):
            x = xytheta_list[i][0]
            y = xytheta_list[i][1]
            theta = xytheta_list[i][2]
            edge_info_list1.append(
                'EDGE_SE2 {} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(i,
                                                                                                         (i + 1) % 12,
                                                                                                         x,
                                                                                                         y, theta, 1, 0,
                                                                                                         0,
                                                                                                         1, 0, 1))

        edge_info_list2 = list()
        xytheta_list = read_data('./CS284_hw2_data/T_everytwo.txt')
        for i in range(len(xytheta_list)):
            x = xytheta_list[i][0]
            y = xytheta_list[i][1]
            theta = xytheta_list[i][2]
            edge_info_list1.append(
                'EDGE_SE2 {} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(i,
                                                                                                         (i + 2) % 12,
                                                                                                         x,
                                                                                                         y, theta, 1, 0,
                                                                                                         0,
                                                                                                         1, 0, 1))

        with open('g2o_data/task3_g2o.g2o', 'w') as f:
            f.writelines(vertex_info_list)
            f.writelines(edge_info_list1)
            f.writelines(edge_info_list2)

    g2o_optimize('g2o_data/task3_g2o.g2o', 'g2o_data/res_task3.g2o')

    end_time = time.time()
    print('Time consumed: {}'.format(end_time - start_time))


def test():
    g2o_optimize('g2o_data/input_MITb_g2o.g2o', 'g2o_data/output_MITb_g2o.g2o')


def check_result(index=2):
    print('-'*15+'Original Error'+'-'*15)
    ref = ref1
    points = [g2o2xytheta(file_path='g2o_data/task{}_g2o.g2o'.format(str(index)), point_index=i) for i in range(12)]
    dis = np.abs(np.array(points) - np.array(ref))
    print('original x error sum: ', dis[:, 0].sum())
    print('original y error sum', dis[:, 1].sum())
    print('original theta error sum', dis[:, 2].sum())

    print('\n'+'-' * 15 + 'Optimized Error' + '-' * 15)
    ref = ref2
    points = [g2o2xytheta(file_path='g2o_data/res_task{}.g2o'.format(str(index)), point_index=i) for i in range(12)]
    dis = np.abs(np.array(points) - np.array(ref))
    print('optimized x error sum: ', dis[:, 0].sum())
    print('optimized y error sum', dis[:, 1].sum())
    print('optimized theta error sum', dis[:, 2].sum())


if __name__ == '__main__':
    if parser.parse_args().task_index == '0':
        test()
    elif parser.parse_args().task_index == '1':
        task1()
    elif parser.parse_args().task_index == '2':
        task2()
        check_result(2)
    elif parser.parse_args().task_index == '3':
        task3()
        check_result(3)

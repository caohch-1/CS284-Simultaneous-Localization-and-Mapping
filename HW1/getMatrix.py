import joblib
import numpy as np
from main import get_depth, depth2coor
from matplotlib import pyplot as plt


def draw_unICP(coor1_x: np.array, coor1_y: np.array, coor2_x: np.array, coor2_y: np.array, index: int = None):
    assert len(coor1_x) == len(coor2_x) == len(coor1_y) == len(coor2_y)
    frame1 = plt.scatter(coor1_x, coor1_y, s=1, c='red')
    frame2 = plt.scatter(coor2_x, coor2_y, s=1, c='green')

    for j in range(len(coor1_x)):
        plt.plot([coor2_x[j], coor1_x[j]], [coor2_y[j], coor1_y[j]], c='g', linewidth=0.4)

    plt.legend([frame1, frame2], ['Old', 'Current'], scatterpoints=2, loc='upper left')
    if index is None:
        plt.title("test")
    else:
        plt.title("Frame{}&{}_unICP".format(str(index), str(index - 1)))
    plt.show()


# para1 Init(frame0) para2 Source(frame index+1) para ICPed
def draw(coor1_x: np.array, coor1_y: np.array, coor2_x: np.array, coor2_y: np.array, coor3_x: np.array,
         coor3_y: np.array, index_: int):
    frame1 = plt.scatter(coor1_x, coor1_y, s=1, c='red')
    frame2 = plt.scatter(coor2_x, coor2_y, s=1, c='green')
    frame3 = plt.scatter(coor3_x, coor3_y, s=1, c='blue')

    plt.legend([frame1, frame2, frame3], ['Init', 'UnICP_frame{}'.format(index_), 'ICPed_frame{}'.format(index_)], scatterpoints=3, loc='upper left')
    plt.show()


def readMatrix(path='./TransformMatrix/1&0'):
    m = joblib.load(path)
    return m


def readMatrixByIndex(index_: int = 1):
    m = joblib.load('./TransformMatrix/{}&{}'.format(index_, index_ - 1))
    return m


if __name__ == '__main__':
    index = 1
    matrixs = list()
    for i in range(index):
        matrixs.append(readMatrixByIndex(i+1))

    depth_data = get_depth("hw1_data/data.txt")
    coordinate_data = depth2coor(depth_data)

    # Init frame
    left0_x = [coordinate_data[0][p][0] for p in range(360)]
    left0_y = [coordinate_data[0][p][1] for p in range(360)]

    # Source frame
    lefts_x = [coordinate_data[index][p][0] for p in range(360)]
    lefts_y = [coordinate_data[index][p][1] for p in range(360)]

    # Current frame
    left2_x = [coordinate_data[index][p][0] for p in range(360)]
    left2_y = [coordinate_data[index][p][1] for p in range(360)]

    for j in range(360):
        for i in range(index):
            left2_x[j], left2_y[j], _ = np.array(np.matmul(matrixs[-1-i], np.mat([left2_x[j], left2_y[j], 1]).T).T)[0]

    draw(left0_x, left0_y, lefts_x, lefts_y, left2_x, left2_y, index)

import numpy as np
from matplotlib import pyplot as plt


# Read Data
def get_depth(path: str = "hw1_data/data.txt"):
    depth_data = open("hw1_data/data.txt")
    depth_data = depth_data.readlines()
    depth_data = [data.split('\t')[:-1] for data in depth_data]
    depth_data = np.array(depth_data, dtype=np.float64)
    return depth_data


# Turn depth data into coordinate
def depth2coor(depth_data: np.array):
    coordinate_data = np.array(
        [[(np.cos(j * np.pi / 180) * depth_data[i][j], np.sin(j * np.pi / 180) * depth_data[i][j]) for j in range(360)]
         for
         i in range(360)], dtype=np.float64)
    return coordinate_data


def de_coor2xy(coor: np.array):
    coordinate_data_x = np.array([[coor[i][j][0] for j in range(360)] for i in range(360)], dtype=np.float64)
    coordinate_data_y = np.array([[coor[i][j][1] for j in range(360)] for i in range(360)], dtype=np.float64)
    return coordinate_data_x, coordinate_data_y


# Calculate nearest point between 2 frames
def cal_nearest(index: int, coor: np.array, coor_x: np.array = None, coor_y: np.array = None):
    nearest_point = list()
    if coor_x is None and coor_y is None:
        coor_x, coor_y = de_coor2xy(coor)

    for j in range(360):
        dis = np.sqrt(np.square(coor_x[index + 1][j] - coor_x[0]) + np.square(coor_y[index + 1][j] - coor_y[0]))
        nearest_point.append((j, np.argmin(dis), np.min(dis)))

    return nearest_point


# # Todo: Here just try 1st and 2rd frame
# # Calculate Rotation and Translation Matrix
# coordinate_data = np.array([[[coordinate_dataX[i][j], coordinate_dataY[i][j], 1] for j in range(360)] for i in range(360)])
# p_avg = np.array([np.average(coordinate_dataX[0]), np.average(coordinate_dataY[0]), 1])
# q_avg = np.array([np.average(coordinate_dataX[1]), np.average(coordinate_dataY[1]), 1])


# Draw
def draw(coor1_x: np.array, coor1_y: np.array, coor2_x: np.array, coor2_y: np.array):
    assert len(coor1_x) == len(coor2_x) == len(coor1_y) == len(coor2_y)
    frame1 = plt.scatter(coor1_x, coor1_y, s=1, c='red')
    frame2 = plt.scatter(coor2_x, coor2_y, s=1, c='green')

    for i in range(len(coor1_x)):
        plt.plot([coor2_x[i], coor1_x[i]], [coor2_y[i], coor1_y[i]], c='g', linewidth=0.4)

    plt.legend([frame1, frame2], ['Old', 'New'], scatterpoints=2, loc='upper left')
    plt.savefig("test.jpg", dpi=1000)
    plt.show()


def rejection(nearest: np.array, method='median'):
    median = np.median(np.array([p[2] for p in nearest]))
    new_nearest = [p for p in nearest if p[2] < 3 * median]
    return new_nearest


if __name__ == '__main__':
    depth_data = get_depth()
    coordinate_data = depth2coor(depth_data)
    for i in range(1):
        # match
        nearest = cal_nearest(i, coordinate_data)
        # reject
        nearest = rejection(nearest)
        print(nearest)
        left1_x = [coordinate_data[i][p[1]][0] for p in nearest]
        left1_y = [coordinate_data[i][p[1]][1] for p in nearest]
        left2_x = [coordinate_data[i+1][p[0]][0] for p in nearest]
        left2_y = [coordinate_data[i+1][p[0]][1] for p in nearest]
        draw(left1_x, left1_y, left2_x, left2_y)



import numpy as np
from matplotlib import pyplot as plt


# Read Data
def get_depth(path: str = "hw1_data/data.txt"):
    depth_data_ = open("hw1_data/data.txt")
    depth_data_ = depth_data_.readlines()
    depth_data_ = [data.split('\t')[:-1] for data in depth_data_]
    depth_data_ = np.array(depth_data_, dtype=np.float64)
    return depth_data_


# Turn depth data into coordinate
def depth2coor(depth_data_: np.array):
    coordinate_data_ = np.array(
        [[(np.cos(j * np.pi / 180) * depth_data_[h][j], np.sin(j * np.pi / 180) * depth_data_[h][j]) for j in
          range(360)]
         for
         h in range(360)], dtype=np.float64)
    return coordinate_data_


def de_coor2xy(coor: np.array):
    coordinate_data_x = np.array([[coor[i][j][0] for j in range(360)] for i in range(360)], dtype=np.float64)
    coordinate_data_y = np.array([[coor[i][j][1] for j in range(360)] for i in range(360)], dtype=np.float64)
    return coordinate_data_x, coordinate_data_y


# Calculate nearest point between 2 frames,
# Return list with (current_frame_index, last_frame_index, distance)
def cal_nearest(index: int, coor: np.array, coor_x: np.array = None, coor_y: np.array = None):
    nearest_point = list()
    if coor_x is None and coor_y is None:
        coor_x, coor_y = de_coor2xy(coor)

    for j in range(360):
        dis = np.sqrt(np.square(coor_x[index + 1][j] - coor_x[0]) + np.square(coor_y[index + 1][j] - coor_y[0]))
        nearest_point.append((j, np.argmin(dis), np.min(dis)))

    return nearest_point


# Draw
def draw(coor1_x: np.array, coor1_y: np.array, coor2_x: np.array, coor2_y: np.array):
    assert len(coor1_x) == len(coor2_x) == len(coor1_y) == len(coor2_y)
    frame1 = plt.scatter(coor1_x, coor1_y, s=1, c='red')
    frame2 = plt.scatter(coor2_x, coor2_y, s=1, c='green')

    for j in range(len(coor1_x)):
        plt.plot([coor2_x[j], coor1_x[j]], [coor2_y[j], coor1_y[j]], c='g', linewidth=0.4)

    plt.legend([frame1, frame2], ['Old', 'Current'], scatterpoints=2, loc='upper left')
    plt.savefig("test.jpg", dpi=1000)
    plt.show()


# Return list with (current_frame_index, last_frame_index, distance)
def rejection(nearest_: np.array, method='median'):
    new_nearest = list()
    if method == 'median':
        median = np.median(np.array([p[2] for p in nearest_]))
        new_nearest = [p for p in nearest_ if p[2] < 3 * median]
    return new_nearest


if __name__ == '__main__':
    depth_data = get_depth()
    coordinate_data = depth2coor(depth_data)
    Rs = list()
    ts = list()
    for i in range(7):
        print('-'*50)
        coordinate_data = depth2coor(depth_data)
        last3loss = list()
        best_loss = np.inf
        bestR = np.zeros((2, 2))
        bestT = np.zeros((2, 1))
        while True:
            # match
            nearest = cal_nearest(i, coordinate_data)
            # reject
            nearest = rejection(nearest)
            # Last frame
            left1_x = [coordinate_data[i][p[1]][0] for p in nearest]
            left1_y = [coordinate_data[i][p[1]][1] for p in nearest]
            left1_t = [0 for i in range(len(nearest))]
            # Current frame
            left2_x = [coordinate_data[i + 1][p[0]][0] for p in nearest]
            left2_y = [coordinate_data[i + 1][p[0]][1] for p in nearest]
            left2_t = [0 for i in range(len(nearest))]
            # Compute the centers of both point clouds
            p = np.mat([np.average(left1_x), np.average(left1_y)], dtype=np.float64)  # Last frame
            q = np.mat([np.average(left2_x), np.average(left2_y)], dtype=np.float64)  # Current frame
            # Compute the matrix
            Q = np.zeros((2, 2), dtype=np.float64)
            for j in range(len(nearest)):
                Q += np.matmul(np.mat([left2_x[j], left2_y[j]]).T - q.T, np.mat([left1_x[j], left1_y[j]]) - p)
            U, _, V_T = np.linalg.svd(Q)
            # Rotation
            R = np.matmul(V_T, U.T)
            # R = np.matmul(U, V_T.T)
            # Translation
            t = p.T - np.matmul(R, q.T)

            for j in range(len(nearest)):
                left2_x[j], left2_y[j] = np.array((np.matmul(R, np.mat([left2_x[j], left2_y[j]]).T) + t).T)[0]
                coordinate_data[i + 1][nearest[j][0]][0] = left2_x[j]
                coordinate_data[i + 1][nearest[j][0]][1] = left2_y[j]

            # Compute loss
            loss = np.sum(np.abs(np.array(left1_x)-np.array(left2_x)) + np.abs(np.array(left1_y)-np.array(left2_y)))

            if best_loss > loss:
                bestT = t
                bestR = R
                best_loss = loss

            if len(last3loss) < 3:
                last3loss.append(loss)
            else:
                last3loss[0] = last3loss[1]
                last3loss[1] = last3loss[2]
                last3loss[2] = loss
                # Terminate
                if (last3loss[2] <= last3loss[1] <= last3loss[0] and abs(last3loss[0] - last3loss[1]) + abs(last3loss[1] - last3loss[2]) < 0.01) \
                        or last3loss[2] >= last3loss[1] >= last3loss[0]:
                    print('Convergence: ', best_loss)
                    draw(left1_x, left1_y, left2_x, left2_y)
                    break

            print('Loss: ', loss)
            # draw(left1_x, left1_y, left2_x, left2_y)
        Rs.append(bestR)
        ts.append(bestT)


import numpy as np
from matplotlib import pyplot as plt
import joblib
import copy


# Read Data
def get_depth(path: str = "hw1_data/data.txt"):
    depth_data_ = open(path)
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

    # if index - 20 < 0:
    #     for h in range(index + 20):
    #         dis = np.sqrt(
    #             np.square(coor_x[index + 1][h] - coor_x[index]) + np.square(coor_y[index + 1][h] - coor_y[index]))
    #         nearest_point.append((h, np.argmin(dis), np.min(dis)))
    #     for h in range((index - 20) % 360, 360):
    #         dis = np.sqrt(
    #             np.square(coor_x[index + 1][h] - coor_x[index]) + np.square(
    #                 coor_y[index + 1][h] - coor_y[index]))
    #         nearest_point.append((h, np.argmin(dis), np.min(dis)))
    # if index + 20 > 359:
    #     for h in range(index - 20, 360):
    #         dis = np.sqrt(
    #             np.square(coor_x[index + 1][h] - coor_x[index]) + np.square(coor_y[index + 1][h] - coor_y[index]))
    #         nearest_point.append((h, np.argmin(dis), np.min(dis)))
    #     for h in range((index + 20) % 360):
    #         dis = np.sqrt(
    #             np.square(coor_x[index + 1][h] - coor_x[index]) + np.square(
    #                 coor_y[index + 1][h] - coor_y[index]))
    #         nearest_point.append((h, np.argmin(dis), np.min(dis)))

    # Special t=360
    for h in range(360):
        dis = np.sqrt(np.square(coor_x[index + 1][h] - coor_x[index]) + np.square(coor_y[index + 1][h] - coor_y[index]))
        nearest_point.append((h, np.argmin(dis), np.min(dis)))

    return nearest_point


# Draw
def draw(coor1_x: np.array, coor1_y: np.array, coor2_x: np.array, coor2_y: np.array, index: int = None):
    assert len(coor1_x) == len(coor2_x) == len(coor1_y) == len(coor2_y)
    frame1 = plt.scatter(coor1_x, coor1_y, s=1, c='red')
    frame2 = plt.scatter(coor2_x, coor2_y, s=1, c='green')

    for j in range(len(coor1_x)):
        plt.plot([coor2_x[j], coor1_x[j]], [coor2_y[j], coor1_y[j]], c='g', linewidth=0.4)

    plt.legend([frame1, frame2], ['Old', 'Current'], scatterpoints=2, loc='upper left')
    if index is None:
        plt.title("test")
        plt.savefig("seq_frame_pics/test.jpg", dpi=1000)
    else:
        plt.title("Frame{}&{}".format(str(index), str(index - 1)))
        plt.savefig("seq_frame_pics/Frame{}&{}.jpg".format(str(index), str(index - 1)))
    # plt.show()
    plt.pause(0.5)


def draw_unICP(coor1_x: np.array, coor1_y: np.array, coor2_x: np.array, coor2_y: np.array, index: int = None):
    assert len(coor1_x) == len(coor2_x) == len(coor1_y) == len(coor2_y)
    frame1 = plt.scatter(coor1_x, coor1_y, s=1, c='red')
    frame2 = plt.scatter(coor2_x, coor2_y, s=1, c='green')

    for j in range(len(coor1_x)):
        plt.plot([coor2_x[j], coor1_x[j]], [coor2_y[j], coor1_y[j]], c='g', linewidth=0.4)

    plt.legend([frame1, frame2], ['Old', 'Current'], scatterpoints=2, loc='upper left')
    if index is None:
        plt.title("test")
        plt.savefig("seq_frame_pics/test.jpg", dpi=1000)
    else:
        plt.title("Frame{}&{}_unICP".format(str(index), str(index - 1)))
        plt.savefig("seq_frame_pics/Frame{}&{}_unICP.jpg".format(str(index), str(index - 1)))
    # plt.show()
    plt.pause(0.5)


# Return list with (current_frame_index, last_frame_index, distance)
def rejection(nearest_: np.array, method='Median'):
    new_nearest = list()
    if method == 'Median':
        median = np.median(np.array([p[2] for p in nearest_]))
        new_nearest = [p for p in nearest_ if p[2] < 3 * median]
    elif method == 'None':
        new_nearest = nearest_
    elif method == 'Trim':
        new_nearest = sorted(nearest_, key=lambda x: x[2], reverse=True)
        new_nearest = new_nearest[int(0.5 * len(nearest_)):]
    elif method == 'Fix':
        new_nearest = [p for p in nearest_ if p[2] < 0.1]

    return new_nearest


if __name__ == '__main__':
    depth_data = get_depth("hw1_data/data.txt")
    coordinate_data = depth2coor(depth_data)
    Ts = list()
    car = [np.mat([1, 0, 1]).T]
    for i in range(359):
        print('-' * 50)
        print(i)
        coordinate_data = depth2coor(depth_data)
        last3loss = list()
        best_loss = np.inf
        bestT = np.identity(3)
        have_draw_unICP = False
        iter_num = 0

        init_left2_x = None
        init_left2_y = None

        while True:
            iter_num += 1
            # match
            old_nearest = cal_nearest(i, coordinate_data)
            # reject
            # if 25 < i % 180 < 75 or 125 < i % 180 < 175:
            #     nearest = rejection(old_nearest, 'None')
            # else:
            #     nearest = rejection(old_nearest, 'Trim')
            nearest = rejection(old_nearest, 'Median')
            # Last frame
            left1_x = [coordinate_data[i][p[1]][0] for p in nearest]
            left1_y = [coordinate_data[i][p[1]][1] for p in nearest]
            # Current frame
            left2_x = [coordinate_data[i + 1][p[0]][0] for p in nearest]
            left2_y = [coordinate_data[i + 1][p[0]][1] for p in nearest]

            if not have_draw_unICP:
                init_left2_x = copy.deepcopy(left2_x)
                init_left2_y = copy.deepcopy(left2_y)

                draw_unICP(left1_x, left1_y, left2_x, left2_y, i + 1)
                have_draw_unICP = True
                print('Init Loss: {}'.format(np.sum(
                    np.abs(np.array(left1_x) - np.array(left2_x)) + np.abs(np.array(left1_y) - np.array(left2_y)))))

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
            # Total Homo matrix
            T = np.zeros((3, 3))
            T[:2, :2] = R
            T[2, :] = np.array([0, 0, 1])
            T[0][2] = t[0]
            T[1][2] = t[1]
            bestT = np.matmul(T, bestT)

            for j in range(len(nearest)):
                left2_x[j], left2_y[j] = np.array((np.matmul(R, np.mat([left2_x[j], left2_y[j]]).T) + t).T)[0]

            for j in range(360):
                coordinate_data[i + 1][j][0], coordinate_data[i + 1][j][1] = \
                    np.array(
                        (np.matmul(R, np.mat([coordinate_data[i + 1][j][0], coordinate_data[i + 1][j][1]]).T) + t).T)[
                        0]

            # Compute loss
            # loss = np.sum(np.abs(np.array(left1_x) - np.array(left2_x)) + np.abs(np.array(left1_y) - np.array(left2_y)))
            loss = np.sum(np.sqrt(np.square(np.array(left1_x) - np.array(left2_x)) + np.square(np.array(left1_y) -
                                                                                               np.array(left2_y))))

            best_loss = min(best_loss, loss)

            # Judge if terminate
            if len(last3loss) < 3:
                last3loss.append(loss)
            else:
                last3loss[0] = last3loss[1]
                last3loss[1] = last3loss[2]
                last3loss[2] = loss
                # Terminate
                if (abs(last3loss[0] - last3loss[1]) + abs(last3loss[1] - last3loss[2]) < 0.001) \
                        or last3loss[2] >= last3loss[1] >= last3loss[0] or iter_num == 30:
                    print('Convergence: ', best_loss)
                    print('Iter num: ', iter_num)
                    Ts.append(bestT)
                    # print(bestT)

                    joblib.dump(bestT, './TransformMatrix/{}&{}'.format(str(i + 1), str(i)))
                    draw(left1_x, left1_y, left2_x, left2_y, i + 1)
                    break

import numpy as np
from matplotlib import pyplot as plt

# Read Data
depth_data = open("hw1_data/data.txt")
depth_data = depth_data.readlines()
depth_data = [data.split('\t')[:-1] for data in depth_data]
depth_data = np.array(depth_data, dtype=np.float64)

# Turn depth data into coordinate
coordinate_dataX = np.array([[np.cos(j * np.pi / 180) * depth_data[i][j] for j in range(360)] for i in range(360)])
coordinate_dataY = np.array([[np.sin(j * np.pi / 180) * depth_data[i][j] for j in range(360)] for i in range(360)])

# Calculate nearest point between 2 frames
nearst_point = [[] for i in range(360)]
for i in range(359):
    for j in range(360):
        dis = np.sqrt(np.square(coordinate_dataX[i+1][j]-coordinate_dataX[i])+np.square(coordinate_dataY[i+1][j]-coordinate_dataY[i]))
        nearst_point[i].append(np.argmin(dis))

# Todo: Here just try 1st and 2rd frame
# Calculate Rotation and Translation Matrix
coordinate_data = np.array([[[coordinate_dataX[i][j], coordinate_dataY[i][j], 1] for j in range(360)] for i in range(360)])
p_avg = np.array([np.average(coordinate_dataX[0]), np.average(coordinate_dataY[0]), 1])
q_avg = np.array([np.average(coordinate_dataX[1]), np.average(coordinate_dataY[1]), 1])


# Draw
frame1 = plt.scatter(coordinate_dataX[0], coordinate_dataY[0], s=1, c='red')
frame2 = plt.scatter(coordinate_dataX[1], coordinate_dataY[1], s=1, c='green')
frame3 = plt.scatter(coordinate_dataX[2], coordinate_dataY[2], s=1, c='blue')
frame4 = plt.scatter(coordinate_dataX[3], coordinate_dataY[3], s=1, c='black')
frame5 = plt.scatter(coordinate_dataX[4], coordinate_dataY[4], s=1, c='pink')

for i in range(360):
    plt.plot([coordinate_dataX[1][i], coordinate_dataX[0][nearst_point[0][i]]], [coordinate_dataY[1][i], coordinate_dataY[0][nearst_point[0][i]]], c='r', linewidth=0.4)
    plt.plot([coordinate_dataX[2][i], coordinate_dataX[1][nearst_point[1][i]]], [coordinate_dataY[2][i], coordinate_dataY[1][nearst_point[1][i]]], c='green', linewidth=0.4)
    plt.plot([coordinate_dataX[3][i], coordinate_dataX[2][nearst_point[2][i]]], [coordinate_dataY[3][i], coordinate_dataY[2][nearst_point[2][i]]], c='blue', linewidth=0.4)
    plt.plot([coordinate_dataX[4][i], coordinate_dataX[3][nearst_point[3][i]]], [coordinate_dataY[4][i], coordinate_dataY[3][nearst_point[3][i]]], c='black', linewidth=0.4)

plt.legend([frame1, frame2, frame3, frame4, frame5], [str(i) for i in range(5)], scatterpoints=5, loc='upper left')
plt.savefig("test.jpg", dpi=1000)
plt.show()
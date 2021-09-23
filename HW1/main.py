import numpy as np
from matplotlib import pyplot as plt

# Read Data
depth_data = open("hw1_data/data.txt")
depth_data = depth_data.readlines()
depth_data = [data.split('\t')[:-1] for data in depth_data]
depth_data = np.array(depth_data, dtype=np.float64)

# Turn depth data into coordinate
coordinate_dataX = np.array([[np.cos((j+1)*np.pi/180)*depth_data[i][j] for j in range(360)] for i in range(360)])
coordinate_dataY = np.array([[np.sin((j+1)*np.pi/180)*depth_data[i][j] for j in range(360)] for i in range(360)])

# Calculate nearest point between 2 frames
nearst_point = [[] for i in range(360)]
for i in range(359):
    for j in range(360):
        dis = np.sqrt(np.square(coordinate_dataX[i+1][j]-coordinate_dataX[i])+np.square(coordinate_dataY[i+1][j]-coordinate_dataY[i]))
        nearst_point[i].append(np.argmin(dis))

# Todo: Here just try 1st and 2rd frame
# Calculate Rptation and Translation Matrix
coordinate_data = np.array([[[coordinate_dataX[i][j], coordinate_dataY[i][j], 1] for j in range(360)] for i in range(360)])
p_avg = np.array([np.average(coordinate_dataX[0]), np.average(coordinate_dataY[0]), 1])
q_avg = np.array([np.average(coordinate_dataX[1]), np.average(coordinate_dataY[1]), 1])


# Draw
frame1 = plt.scatter(coordinate_dataX[0], coordinate_dataY[0], s=1, c='r')
frame2 = plt.scatter(coordinate_dataX[1], coordinate_dataY[1], s=1, c='g')

for i in range(360):
    plt.plot([coordinate_dataX[1][i], coordinate_dataX[0][nearst_point[0][i]]], [coordinate_dataY[1][i], coordinate_dataY[0][nearst_point[0][i]]], c='b', linewidth=0.6)

plt.legend([frame1, frame2], ['Old', 'New'], scatterpoints=1)
plt.show()
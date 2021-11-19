import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage import measure

K = np.array([[259.2, 0, 160],
              [0, 259.2, 120],
              [0, 0, 1]])
disparity = 1.3476e5
SIDE_LENGTH = 0.2
SIDE_LENGTH_UNIT = 0.01
VOXEL_NUM = int(pow(SIDE_LENGTH / SIDE_LENGTH_UNIT, 3))
TRUNCATED_DIS = 1


def read_glb2cam():
    root_path = './CS284_hw5_data/pose/'
    glb2cam_list = list()
    for pose_file in os.listdir(root_path):
        with open(root_path + pose_file, 'r') as f:
            matrix = f.readlines()
            matrix = [row.strip().split(',') for row in matrix]
            matrix = [[float(ele) for ele in row] for row in matrix]
            glb2cam_list.append(np.array(matrix, dtype=np.float32))
    return glb2cam_list


def read_depth_image():
    root_path = './CS284_hw5_data/depth/'
    depth_list = list()
    for depth_file in os.listdir(root_path):
        depth_list.append(cv2.cvtColor(cv2.imread(root_path + depth_file), cv2.COLOR_RGB2GRAY))
    return depth_list


class Frame(object):
    def __init__(self, img, glb2cam):
        self.rawDepthImg = img
        self.glb2cam = glb2cam
        self.trueDepth = self._get_true_depth()

    def _get_true_depth(self):
        return self.rawDepthImg * K[0][0] / disparity * 100


class KinectFusion(object):
    def __init__(self):
        self.frameId = -1
        self.frameList = list()

        self.truncationDis = TRUNCATED_DIS

        self.cameraIntrinsic = K
        self.inverseCameraIntrinsic = self._inverse_camera_intrinsic()

        self.voxelLength = 0.2  # meter
        self.voxelLengthUnit = 0.01  # meter
        self.voxelNum = int(self.voxelLength / self.voxelLengthUnit)  # One side, pow(#, 3) to get all
        self.tsdfMat = np.full((self.voxelNum, self.voxelNum, self.voxelNum), None)
        self.tsdfWeightMat = np.zeros((self.voxelNum, self.voxelNum, self.voxelNum))

    def add_frame(self, frame: Frame):
        self.frameId += 1
        self.frameList.append(frame)

    def _get_lambda(self, u, v):
        la = self.inverseCameraIntrinsic @ np.array([u, v, 1]).T
        la = np.linalg.norm(la)
        return la

    def _inverse_camera_intrinsic(self):
        f = self.cameraIntrinsic[0][0]
        cx = self.cameraIntrinsic[0][2]
        cy = self.cameraIntrinsic[1][2]
        return (1 / f) * np.array([[1, 0, -cx],
                                   [0, 1, -cy],
                                   [0, 0, f]])

    """Todo: Whether here true? 100 need?
        Something wrong here
    """
    def _voxel2glb(self, x, y, z):
        xg, yg, zg = np.array([x, y, z]) * self.voxelLengthUnit * 100
        return xg, yg, zg

    def _glb2cam(self, xg, yg, zg):
        glb2cam = self.frameList[self.frameId].glb2cam
        camCoor = glb2cam @ np.array([xg, yg, zg, 1]).T
        xc, yc, zc = (camCoor / camCoor[3])[:3]
        return xc, yc, zc

    def _cam2img(self, xc, yc, zc):
        uv = self.cameraIntrinsic @ np.array([xc, yc, zc]).T
        uv = (uv / uv[2])[:2]
        return np.around(uv).astype(int)

    def _get_mesh(self):
        tsdfVolume = self.tsdfMat
        tsdfVolume[tsdfVolume is None] = 1
        verts, faces, _, _ = measure.marching_cubes(tsdfVolume)
        return verts, faces

    def vis(self):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        verts, faces = self._get_mesh()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(verts[faces])
        face_color = [0.5, 0.5, 1]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

        # ax.set_xlim(-20, 20)
        # ax.set_ylim(0, 320)
        # ax.set_zlim(0, 3)

        plt.show()

    def run(self):
        print('='*30+str(self.frameId)+'='*30)
        for i in range(int(-self.voxelNum / 2), int(self.voxelNum / 2) + 1):
            for j in range(int(-self.voxelNum / 2), int(self.voxelNum / 2) + 1):
                for k in range(int(-self.voxelNum / 2), int(self.voxelNum / 2) + 1):
                    xg, yg, zg = self._voxel2glb(i, j, k)
                    xc, yc, zc = self._glb2cam(xg, yg, zg)
                    u, v = self._cam2img(xc, yc, zc)
                    if not (0 <= u < 240 and 0 <= v < 320):
                        continue

                    lam = self._get_lambda(u, v)
                    rawDepth = self.frameList[self.frameId].trueDepth[u][v]
                    if rawDepth == 0:
                        continue

                    sdf = 1 / lam * np.linalg.norm(np.array([xc, yc, zc])) - rawDepth
                    sdf *= -1
                    tsdf = min(1, sdf / self.truncationDis) if sdf >= - self.truncationDis else None
                    print(1 / lam * np.linalg.norm(np.array([xc, yc, zc])), rawDepth, sdf)
                    if tsdf is None:
                        continue

                    # update fusion
                    currTsdf = tsdf
                    currWeight = 1

                    preTsdf = self.tsdfMat[i][j][k]
                    preWeight = self.tsdfWeightMat[i][j][k]

                    if preTsdf is None:
                        newTsdf = currTsdf
                        newWeight = currWeight
                    else:
                        newTsdf = (preWeight * preTsdf + currWeight * currTsdf) / (preWeight + currWeight)
                        newWeight = preWeight + currWeight

                    self.tsdfMat[i][j][k] = newTsdf
                    self.tsdfWeightMat[i][j][k] = newWeight


if __name__ == '__main__':
    depthList = read_depth_image()
    glb2camList = read_glb2cam()
    kf = KinectFusion()
    for i_ in range(len(depthList)):
        kf.add_frame(Frame(depthList[i_], glb2camList[i_]))
        kf.run()
        print()
    # kf.vis()
    # for i in range(kf.tsdfMat.shape[0]):
    #     for j in range(kf.tsdfMat.shape[1]):
    #         for k in range(kf.tsdfMat.shape[2]):
    #             print((i, j, k), ': ', kf.tsdfMat[i][j][k])

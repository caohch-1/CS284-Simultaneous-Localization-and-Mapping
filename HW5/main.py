import time

import numpy as np
import os
import cv2
from skimage import measure


def makePlyFile(xyzs, fileName='res.ply'):
    with open(fileName, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment PCL generated\n')
        f.write('element vertex {}\n'.format(len(xyzs)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyzs)):
            x, y, z = xyzs[i]
            f.write('{} {} {}\n'.format(x, y, z))


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
        depth_list.append(cv2.imread(root_path + depth_file, -1))
    return depth_list


K = np.array([[259.2, 0, 160],
              [0, 259.2, 120],
              [0, 0, 1]])
disparity = 1.3476e5
SIDE_LENGTH = 0.2
SIDE_LENGTH_UNIT = 0.002
VOXEL_NUM = int(pow(SIDE_LENGTH / SIDE_LENGTH_UNIT, 3))
TRUNCATED_DIS = SIDE_LENGTH_UNIT / 2


class Frame(object):
    def __init__(self, img, glb2cam):
        self.rawDepthImg = img.astype(np.float)
        self.glb2cam = glb2cam
        self.glb2cam[:3, 3] = self.glb2cam[:3, 3] * 100
        self.trueDepth = self._get_true_depth()

    def _get_true_depth(self):
        # return self.rawDepthImg * K[0][0] / disparity * 100
        res = self.rawDepthImg / disparity * 100
        for u in range(self.rawDepthImg.shape[0]):
            for v in range(self.rawDepthImg.shape[1]):
                res[u, v] = ((np.linalg.pinv(K) @ np.array([u, v, 1]).T) * res[u, v])[2]
        return res


class KinectFusion(object):
    def __init__(self):
        self.frameId = -1
        self.frameList = list()

        self.truncationDis = TRUNCATED_DIS

        self.cameraIntrinsic = K
        self.inverseCameraIntrinsic = self._inverse_camera_intrinsic()

        self.voxelLength = SIDE_LENGTH  # meter
        self.voxelLengthUnit = SIDE_LENGTH_UNIT  # meter
        self.voxelNum = int(self.voxelLength / self.voxelLengthUnit)  # One side, pow(#, 3) to get all
        self.tsdfMat = np.full((self.voxelNum, self.voxelNum, self.voxelNum), None)
        self.tsdfWeightMat = np.zeros((self.voxelNum, self.voxelNum, self.voxelNum))

    def _get_mesh(self):
        for _i in range(self.voxelNum):
            for _j in range(self.voxelNum):
                for _k in range(self.voxelNum):
                    self.tsdfMat[_i][_j][_k] = 1.0 if self.tsdfMat[_i][_j][_k] is None else self.tsdfMat[_i][_j][_k]

        verts, faces, _, _ = measure.marching_cubes(self.tsdfMat, method='lewiner')
        return verts, faces

    def vis(self):
        import open3d as o3d
        if os.path.exists('./result.ply'):
            pcd = o3d.io.read_point_cloud('./result.ply', format="xyz")
            o3d.visualization.draw_geometries([pcd])
        else:
            print("Didn't save result.")

    def add_frame(self, frame: Frame):
        self.frameId += 1
        self.frameList.append(frame)

    def _inverse_camera_intrinsic(self):
        f = self.cameraIntrinsic[0][0]
        cx = self.cameraIntrinsic[0][2]
        cy = self.cameraIntrinsic[1][2]
        return (1 / f) * np.array([[1, 0, -cx],
                                   [0, 1, -cy],
                                   [0, 0, f]])

    def _get_lambda(self, u, v):
        la = self.inverseCameraIntrinsic @ np.array([u, v, 1]).T
        la = np.linalg.norm(la)
        return la

    def _voxel2glb(self, x, y, z):
        xg, yg, zg = np.array([x + 0.5, y + 0.5, z + 0.5]) * self.voxelLengthUnit * 100 - SIDE_LENGTH * 100 / 2
        return xg, yg, zg

    def _glb2cam(self, xg, yg, zg):
        glb2cam = self.frameList[self.frameId].glb2cam
        camCoor = glb2cam @ np.array([xg, yg, zg, 1])
        xc, yc, zc = camCoor[:3]
        return xc, yc, zc

    def _cam2img(self, xc, yc, zc):
        uv = self.cameraIntrinsic @ np.array([xc, yc, zc])
        uv = (uv / uv[2])[:2]

        return np.floor(uv).astype(int)

    def run(self):
        print('=' * 30 + str(self.frameId) + '=' * 30)
        count = 0
        count_uv = 0
        count_depth = 0
        count_sdf = 0
        for i__ in range(0, self.voxelNum):
            for j__ in range(0, self.voxelNum):
                for k__ in range(0, self.voxelNum):
                    count += 1
                    if count % (pow(self.voxelNum, 3) // 10) == 0 or count == pow(self.voxelNum, 3):
                        print('\rComputing TSDF {:.2f}% ...'.format(100 * count / pow(self.voxelNum, 3)), flush=True)
                    xg, yg, zg = self._voxel2glb(i__, j__, k__)
                    xc, yc, zc = self._glb2cam(xg, yg, zg)
                    u, v = self._cam2img(xc, yc, zc)
                    if not (0 <= u < 240 and 0 <= v < 320):
                        count_uv += 1
                        continue

                    lam = self._get_lambda(u, v)
                    rawDepth = self.frameList[self.frameId].trueDepth[u, v]
                    if rawDepth == 0:
                        count_depth += 1
                        continue

                    sdf = (1 / lam) * np.linalg.norm(
                        np.array([xg, yg, zg]) - np.linalg.pinv(self.frameList[self.frameId].glb2cam)[:3, 3])
                    sdf -= rawDepth

                    if sdf >= -self.truncationDis:
                        tsdf = min(1, sdf / self.truncationDis) * np.sign(sdf)
                    else:
                        count_sdf += 1
                        continue

                    # update fusion
                    currTsdf = tsdf
                    currWeight = 1

                    preTsdf = self.tsdfMat[i__][j__][k__]
                    preWeight = self.tsdfWeightMat[i__][j__][k__]

                    if preTsdf is None:
                        newTsdf = currTsdf
                        newWeight = currWeight
                    else:
                        newTsdf = (preWeight * preTsdf + currWeight * currTsdf) / (preWeight + currWeight)
                        newWeight = preWeight + currWeight

                    self.tsdfMat[i__][j__][k__] = newTsdf
                    self.tsdfWeightMat[i__][j__][k__] = newWeight

                    # if tsdf != 1.0 and tsdf != -1.0:
                    #     print('uv:{}, voxel:{}, SDF:{}, TSDF:{}'.format((u, v), (i__, j__, k__), sdf, tsdf))
        # print('uv:{}, depth:{}, sdf:{}, all:{}, left: {}'.format(count_uv, count_depth, count_sdf, count,
        #                                                          count - count_uv - count_depth - count_sdf))


if __name__ == '__main__':
    st = time.time()
    depthList = read_depth_image()
    glb2camList = read_glb2cam()
    kf = KinectFusion()
    # for i_ in range(len(depthList))[:]:
    #     kf.add_frame(Frame(depthList[i_], glb2camList[i_]))
    #     kf.run()
    #     print()
    kf.vis()
    print('Time:{}'.format(time.time() - st))

    # show_ply('./result_frame/1frame.ply')
    # for i in range(14):
    #     show_ply('./{}.ply'.format(i))
    # for i in range(kf.tsdfMat.shape[0]):
    #     for j in range(kf.tsdfMat.shape[1]):
    #         for k in range(kf.tsdfMat.shape[2]):
    #             print((i, j, k), ': ', kf.tsdfMat[i][j][k])

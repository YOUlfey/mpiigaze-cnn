import os

import cv2 as cv
import numpy as np
import scipy.io as sio
import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='res/data/extract/MPIIGaze/Data/Normalized')
    parser.add_argument('--out', type=str, default='res/data/out.npz')
    return parser.parse_args()


def __read_mat(path_mat):
    content = sio.loadmat(path_mat, struct_as_record=False, squeeze_me=True)
    data = content['data']
    return data


def __convert_pose(vect):
    M, _ = cv.Rodrigues(np.array(vect).astype(np.float32))
    vec = M[:, 2]
    phi = np.arctan2(vec[0], vec[2])
    theta = np.arcsin(vec[1])
    return np.array([theta, phi])


def __convert_gaze(vect):
    x, y, z = vect
    phi = np.arctan2(-x, -z)
    theta = np.arcsin(-y)
    return np.array([theta, phi])


def __get_data(path_data):
    images = []
    poses = []
    gazes = []

    for patient in os.listdir(path_data):
        full_path_patient = os.path.join(path_data, patient)
        for day_name in os.listdir(full_path_patient):
            full_day_path = os.path.join(full_path_patient, day_name)
            print('Read data from: ', full_day_path)

            content = __read_mat(full_day_path)

            left_images = content.left.image
            left_poses = content.left.pose
            left_gazes = content.left.gaze

            right_images = content.right.image
            right_poses = content.right.pose
            right_gazes = content.right.gaze

            if left_images.shape == (36, 60):
                left_images = left_images[np.newaxis, :, :]
                left_gazes = left_gazes[np.newaxis, :]
                left_poses = left_poses[np.newaxis, :]

            if right_images.shape == (36, 60):
                right_images = right_images[np.newaxis, :, :]
                right_gazes = right_gazes[np.newaxis, :]
                right_poses = right_poses[np.newaxis, :]

            for i in np.arange(0, len(left_gazes), 1):

                images.append(left_images[i])
                images.append(right_images[i])

                poses.append(__convert_pose(left_poses[i]))
                poses.append(__convert_pose(right_poses[i]))

                gazes.append(__convert_gaze(left_gazes[i]))
                gazes.append(__convert_gaze(right_gazes[i]))

    return images, poses, gazes


args = parser_args()
img, pss, gzs = __get_data(args.data)
np.savez(args.out, image=img, pose=pss, gaze=gzs)

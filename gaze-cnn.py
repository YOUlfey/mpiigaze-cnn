import json
import os
from datetime import datetime

import numpy as np
from keras.backend import cos, sin, sqrt
from keras.backend import mean
from keras.callbacks import CSVLogger
from keras.models import model_from_json
from keras.optimizers import RMSprop
from tensorflow import acos
import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='res/data/out.npz')
    parser.add_argument('--log', type=str, default='res/log')
    parser.add_argument('--model', type=str, default='res/default-model.json')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=512)
    return parser.parse_args()


def load_data(path):
    data = np.load(path)
    gazes = data['gaze']
    images = data['image']
    poses = data['pose']
    index = int(len(gazes) * 0.5)
    train_images = images[0:index]
    train_gazes = gazes[0:index]
    train_poses = poses[0:index]
    test_images = images[index:len(images)]
    test_gazes = gazes[index:len(gazes)]
    test_poses = poses[index:len(poses)]
    train_images = np.reshape(train_images, (len(train_images), 36, 60, 1))
    train_images = train_images.astype('float32') / 255
    test_images = np.reshape(test_images, (len(test_images), 36, 60, 1))
    test_images = test_images.astype('float32') / 255
    return train_images, train_gazes, train_poses, test_images, test_gazes, test_poses


def convert_gaze(angles):
    x = -cos(angles[:, 0]) * sin(angles[:, 1])
    y = -sin(angles[:, 0])
    z = -cos(angles[:, 1]) * cos(angles[:, 1])
    norm = sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def degrees_mean_error(y_true, y_pred):
    x_p, y_p, z_p = convert_gaze(y_pred)
    x_t, y_t, z_t = convert_gaze(y_true)
    angles = mean(x_p * x_t + y_p * y_t + z_p * z_t)
    return acos(angles) * 180 / np.pi


args = parser_args()
infile = open(args.model, 'r')
obj = json.load(infile)
model = model_from_json(json.dumps(obj))

if not os.path.exists(args.log):
    os.mkdir(args.log)

session_path = os.path.join(args.log, datetime.now().isoformat())
if not os.path.exists(session_path):
    os.mkdir(session_path)

csv_logger = CSVLogger(os.path.join(session_path, 'log_fit.csv'), append=True, separator=';')


with open(os.path.join(session_path, 'scheme.txt'), 'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

model.compile(optimizer=RMSprop(lr=0.0001), loss=degrees_mean_error, metrics=['acc'])

x_train, y_train, x_poses_train, x_test, y_test, x_poses_test = load_data(args.data)

history = model.fit([x_train, x_poses_train], y_train, epochs=args.epochs, batch_size=args.batch, validation_data=([x_test, x_poses_test], y_test), callbacks=[csv_logger])

model.save(os.path.join(session_path, 'gaze-model.h5'))

history = history.history

outfile = open(os.path.join(session_path, 'history'), 'w')
json.dump(history, outfile, indent=2)
outfile.close()

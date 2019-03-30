from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input, Concatenate, Dropout
from keras.models import Model
import os
import json
import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='res/default-model.json')
    return parser.parse_args()


def __get_model():
    image_input = Input((36, 60, 1))
    poses_input = Input((2, ))
    conv_layer = Conv2D(20, (5, 5), activation='relu')(image_input)
    maxpool_layer = MaxPool2D((2, 2))(conv_layer)
    conv_layer = Conv2D(50, (5, 5), activation='relu')(maxpool_layer)
    maxpool_layer = MaxPool2D((2, 2))(conv_layer)
    flatten_layer = Flatten()(maxpool_layer)
    dense_layer = Dense(500, activation='relu')(flatten_layer)
    concat_layer = Concatenate()([dense_layer, poses_input])
    output = Dense(2)(concat_layer)
    return Model(inputs=[image_input, poses_input], outputs=output)


args = parser_args()
model = __get_model()
obj = json.loads(model.to_json())
model_file = open(args.model, 'w')
json.dump(obj, model_file, indent=2)
model_file.close()

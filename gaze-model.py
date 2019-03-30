from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input, Concatenate, Dropout
from keras.models import Model
import os
import json


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


model = __get_model()
json_str = model.to_json()
model_file = open(os.path.join(os.getcwd(), 'res/default-model.json'))
json.dump(json_str, model_file, indent=2)
model_file.close()

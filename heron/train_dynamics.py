# For some reason the venv is not recognizing that these directories are already in PATH
import os
if os.name == 'nt':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/tools/cuda/bin")
    os.add_dll_directory('C:/tools/zlib/dll_x64')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#https://stackoverflow.com/questions/50090173/how-to-give-input-to-the-middle-layer-in-keras
#https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
def create_dynamics_model():
    # Network defined by the Deepmind paper
    encoder_input = layers.Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    e1 = layers.Conv2D(32, 8, strides=4, activation="relu")(encoder_input)
    e2 = layers.Conv2D(64, 4, strides=2, activation="relu")(e1)
    e3 = layers.Conv2D(64, 3, strides=1, activation="relu")(e2)

    # double check these params
    bottleneck = layers.Conv2D(64, 3, strides=1, activation="relu")(e3)

    # Connect each decoder layer to the corresponding encoder layer at the same depth
    d1 = layers.Conv2DTranspose(64, 3, strides=1, activation="relu")(bottleneck)
    d1 = keras.concatenate([d1.output, e3.output])
    # d1 = keras.Concatenate()([d1,  action.output])

    d2 = layers.Conv2DTranspose(64, 4, strides=2, activation="relu")(d1)
    d2 = keras.concatenate([d2.output, e2.output])(d1)

    d3 = layers.Conv2DTranspose(32, 8, strides=4, activation="relu")(d2)
    d3 = keras.concatenate([d3.output, e1.output])(d3)

    output = layers.Conv2DTranspose(3, 4, strides=2, activation='tanh')(d3)
    model = keras.Model(inputs=encoder_input, outputs=output)
    

    # convnet = layers.Flatten()(convnet)

    # # Creating the action input layer
    # action = layers.Input(shape=(1,))

    # combined = keras.concatenate([convnet.output, action.output])

    # z = Dense(2, activation="relu")(combined)
    # z = Dense(1, activation="linear")(z)

    # return keras.Model(inputs=[convnet.input, action.input], outputs=z)
    return model


model = create_dynamics_model()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# TODO: Create training and validation loops

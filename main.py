import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np 
import matplotlib.pyplot as plt 

from bcnn_utils import img_gen



trimg_dir = "/Users/karljaehnig/archive/seg_train/seg_train/"
tsimg_dir = "/Users/karljaehnig/archive/seg_test/seg_test/"

trgen,tsgen = img_gen(trimg_dir,tsimg_dir)

def model():

    mdl = Sequential()

    mdl.add(Conv2D(
        input_shape=(150,150,3), 
        filters=32, 
        kernel_size=(3,3),
        activation='relu',
        padding='same',
        # kernel_initializer='normal'
        )
    )
    mdl.add(MaxPooling2D(2,2))

    # mdl.add(Conv2D(
    #     filters=32, 
    #     kernel_size=(3,3),
    #     activation='relu',
    #     padding='same',
    #     # kernel_initializer='normal'
    #     )
    # )


    # mdl.add(MaxPooling2D(2,2))

    mdl.add(Conv2D(64, (3,3), 
            activation='relu',
            padding='same',
            # kernel_regularizer='l2',
            # kernel_initializer='normal'
            ))

    mdl.add(MaxPooling2D(2,2))
    # mdl.add(BatchNormalization())

    mdl.add(Conv2D(64, (3,3), 
            activation='relu',
            padding='same',
            # kernel_regularizer='l2',
            # kernel_initializer='normal'
            ))

    mdl.add(MaxPooling2D(2,2))
    # mdl.add(BatchNormalization())

    mdl.add(Conv2D(128, (3,3), 
            activation='relu',
            padding='same',
            # kernel_regularizer='l2',
            # kernel_initializer='normal'
            ))

    mdl.add(MaxPooling2D(2,2))
    # mdl.add(BatchNormalization())

    mdl.add(Conv2D(128, (3,3), 
            activation='relu',
            padding='same',
            # kernel_regularizer='l2',
            # kernel_initializer='normal'
            ))

    mdl.add(MaxPooling2D(2,2))
    # mdl.add(BatchNormalization())

    mdl.add(Flatten())

    mdl.add(Dropout(0.75))

    mdl.add(Dense(512, activation='relu'))

    mdl.add(Dropout(0.25))

    mdl.add(Dense(256, activation='relu'))


    mdl.add(
        Dense(units=6, activation='softmax')
        )
 
    return mdl

# mdl = model()

# mdl.summary()

# mdl.compile(
#     loss='categorical_crossentropy',
#     optimizer=Adam(learning_rate=0.005),
#     metrics=['accuracy']
#     )

# history_mdl = mdl.fit(
#     trgen,
#     epochs=100,
#     verbose=1
#     )

# try:
#     history_mdl.save("/Users/karljaehnig/Repositories/bayesian_NN_experiments/bcnn_classifier/saved_models/standby_intel_model")
# except:
#     mdl.save("/Users/karljaehnig/Repositories/bayesian_NN_experiments/bcnn_classifier/saved_models/standby_intel_model")







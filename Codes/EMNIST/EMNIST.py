import os
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
#import random 



from scipy.misc import imread
import matplotlib.pyplot as plt

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


import cv2
from imutils.perspective import four_point_transform
from imutils import contours
from skimage import io
from skimage.feature import hog
import imutils
from skimage.transform import resize, downscale_local_mean
import matplotlib.pyplot as plt
from keras.models import load_model


train=pd.read_csv("G:/identify the taxi numberplate/Alpha/train.csv",index_col=None)
#test=("G:\identify the taxi numberplate\emnist-byclass-tedt.csv",index_col=None)


train_y = train.iloc[:100000,0]
train_y = keras.utils.to_categorical(train_y, 62)


train_x = train.iloc[:100000,1:]
train_x = train_x.astype('float32')
train_x = train_x/255

train_x-=np.mean(train_x,axis=0)
train_x = train_x.values.reshape(-1,28,28,1)


split_size=int(train_x.shape[0]*0.7)

train_X, val_X = train_x[:split_size], train_x[split_size:]
train_Y, val_Y = train_y[:split_size], train_y[split_size:]


def model():
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5),  activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, (1, 1), activation='relu'))
    model.add(Conv2D(32, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    BatchNormalization()
    
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    BatchNormalization()
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (1, 1), activation='relu'))
    model.add(Conv2D(128, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    BatchNormalization()
    

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(62, activation='softmax'))
     
    return model

model1 = model()
epochs = 6
model1.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
 
history = model1.fit(train_X, train_Y,epochs=epochs, verbose=1, 
                   validation_data=(val_X, val_Y))


model1.save("6epochsmodel")




# Returns a compiled model identical to the previous one
model1 = load_model('6epochsmodel')


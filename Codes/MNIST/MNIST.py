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


train=pd.read_csv("G:/identify the digits/kaggle/train.csv", index_col=None)
test=pd.read_csv("G:/identify the digits/kaggle/test.csv", index_col=None)
sub=pd.read_csv("G:/identify the digits/kaggle/sample_submission.csv", index_col=None)

feature=train.pop("label")

train=train/255
test=test/255

temp=[]

train_x = train.values.reshape(-1,28,28,1)
test_y = test.values.reshape(-1,28,28,1)    
         
train_x-=np.mean(train_x,axis=0)
test_y-=np.mean(test_y,axis=0)


split_size=int(train_x.shape[0]*0.7)
train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = feature[:split_size], feature[split_size:]

train_y = keras.utils.to_categorical(train_y, 10)
val_y = keras.utils.to_categorical(val_y, 10)



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
    
    """
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    BatchNormalization()
    """

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
     
    return model

model1 = model()
epochs = 2
model1.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
 
history = model1.fit(train_x, train_y,epochs=epochs, verbose=1, 
                   validation_data=(val_x, val_y))



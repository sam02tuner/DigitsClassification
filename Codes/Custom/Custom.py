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


train=pd.read_csv("G:/identify the digits/Train_UQcUa52/train.csv", index_col=None)
test=pd.read_csv("G:/identify the digits/Test_fCbTej3.csv", index_col=None)

path="G:/identify the digits/Train_UQcUa52/Images/train"

temp=[]
for ig in train.filename:
    filepath=os.path.join(path,ig)
    image=imread(filepath)
    image=image.astype('float32')
    temp.append(image)   
    train_x=np.stack(temp)

train_x-=np.mean(train_x,axis=0)


split_size=int(train_x.shape[0]*0.7)
train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train.label.values[:split_size], train.label.values[split_size:]

train_y = keras.utils.to_categorical(train_y, 10)
val_y = keras.utils.to_categorical(val_y, 10)


def model():
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5),  activation='relu', input_shape=(28,28,4)))
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
epochs = 1
model1.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
 
history = model1.fit(train_x, train_y,epochs=epochs, verbose=1, 
                   validation_data=(val_x, val_y))


path1="G:/identify the digits/Train_UQcUa52/Images/test"

temp1=[]
for ig in test.filename:
    filepath=os.path.join(path1,ig)
    image=imread(filepath)
    image=image.astype('float32')
    temp1.append(image)   
    test_x=np.stack(temp1)

test_x-=np.mean(test_x,axis=0)



labels=model1.predict(test_x)


nlabs=pd.DataFrame(labels)

wer1=pd.DataFrame(columns=["filename","label"])
wer1.filename=test["filename"]

num = nlabs[[0,1,2,3,4,5,6,7,8,9]].max(axis=1)

for i in wer1.index:
    for j in [0,1,2,3,4,5,6,7,8,9]:
        if(nlabs.ix[i,j]==num.ix[i]):
            wer1.ix[i,"label"]=j
            break
        else:
            continue


a=wer1.pop("filename") 
wer1.index=a   
wer1.ffill(inplace=True)
wer1.to_csv("G:/identify the digits/Submissionnew3.csv")



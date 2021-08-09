#importing necessary libraries
import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import layers
from tensorflow import keras


X_train,train_y,X_test,test_y = [],[],[],[]

#reading the dataset
emotions = pd.read_csv('fer2013.csv')
#setting the values in a list for training and testing
for index, row in emotions.iterrows():
    val = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(val,'float32'))
            train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val,'float32'))
            test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")

#values to be used when training the model
num_features = 64
num_labels = 7
batch_size = 64
epochs = 20
width, height = 48, 48

#assigning the variables with numpy arrays for respective training and testing
X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')

train_y =  to_categorical(train_y, num_classes = num_labels)
test_y =  to_categorical(test_y, num_classes = num_labels)

#normalizing data between 0 and 1
X_train -= np.mean(X_train, axis = 0)
X_train /= np.std(X_train, axis = 0)

X_test -= np.mean(X_test, axis = 0)
X_test /= np.std(X_test, axis = 0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

#CNN model creation
model = Sequential()
#first convolutional layer
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', input_shape = (X_train.shape[1:])))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.5))

#second convolutional Layer
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.5))

#third convolutional Layer
model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(Conv2D(128, (3,3), activation = 'relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

model.add(Flatten())

#fully connected layer
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation = 'softmax'))

#compiling the model
model.compile(loss=categorical_crossentropy,
             optimizer='sgd',
             metrics=['accuracy'])

#training the model
model.fit(X_train,train_y,
         batch_size = batch_size,
         epochs = epochs,
         verbose = 1,
         validation_data = (X_test,test_y),
         shuffle = True)

#saving the model for later use
fer_json = model.to_json()
with open("fer.json","w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")

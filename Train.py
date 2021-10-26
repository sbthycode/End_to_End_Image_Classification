# Importing required libraries
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Defining the training path of the images and the .csv file for Labels and Value
train = pd.read_csv('/content/drive/MyDrive/IITD Hack/SoML-50/annotations.csv')
#test = pd.read_excel(r'C:\Users\HP\OneDrive - iitgn.ac.in\Desktop\IITD hack\SoML-50\test.xlsx')
train_folder = '/content/drive/MyDrive/IITD Hack/SoML-50/data'
#test_folder = r'C:\Users\HP\OneDrive - iitgn.ac.in\Desktop\IITD hack\SoML-50\test_imgs'


# Defining data generator
train_gen = ImageDataGenerator(rescale=1./255)
train_data = train_gen.flow_from_dataframe(dataframe=train, directory=train_folder, x_col='Image',
                                           y_col='Label', batch_size=64, shuffle=False,
                                           class_mode='categorical', target_size=(64, 192))

# Creating the Layers of our model with tensorflow's Sequential API
model = Sequential()

# convolutional layers
model.add(Conv2D(50, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(64, 192, 3)))

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(3, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 2 epochs
history = model.fit(train_data, epochs=2, steps_per_epoch=780)

# saving the model as an h5 file to be used later in inference1.py and inference2.py
model.save('Task1Model.h5')

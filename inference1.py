# 
import sys
import os
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#input from the command line
inpu = sys.argv[1]

#generating the absolute path for directory
script_dir1 = os.path.dirname(__file__)
script_dir2 = os.path.split(script_dir1)[0] 
rel_path = inpu
directory = os.path.join(script_dir2, rel_path)

# Loading the saved model 'Task1Model.h5'
model = tf.keras.models.load_model(os.path.join(script_dir2,'Go_deeper\Task1Model.h5'))

#iterating over the test images and creation of the csv file 
a=[]
images=[]
for imag in os.listdir(directory):

    img = tf.keras.preprocessing.image.load_img(os.path.join(directory, imag), target_size=(64, 192))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img_test = img/255.0
    
    pred = model.predict(img_test)
    if (np.argmax(pred)==0):
        a.append('infix')
    elif (np.argmax(pred)==1):
        a.append('postfix')
    elif (np.argmax(pred)==2):
        a.append('prefix')
    images.append(str(imag))

Y_pred=pd.Series(a)
X_pred=pd.Series(images)
frame={ 'Image_Name': X_pred, 'Label': Y_pred }
result = pd.DataFrame(frame)
result.to_csv('Go_deeper_1.csv')




# End_to_End_Image_Classification
The objective of this project was the classification of 384x128 dimensioned images into three classes. 

# Train.py  
We used tensorflow to create a custom deep learning model using complicated CNN architecture. Some of the layers we used include - Conv2D, Dropout, MaxPool2D, Flatten, Dense. We also used ImageDataGenerator and coupled it with flow_from_dataframe to iterate over and label the training dataset. We successfully trained a model with over 99% training accuracy and saved it as an .h5 file.

# inference1.py 
We loaded the previously saved model and used it to evaluate on the specified test folder. As we will be provided with the relative path, we are assuming that the test folder will be created within this submission folder. 

Our code is also creating the .csv file as expected.This project was a part of, and our submission for, the IIT Delhi Hackathon, 2021.

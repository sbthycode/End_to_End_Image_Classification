We, as freshers, thoroughly enjoyed working in this project. Although the hackathon was a really thought provoking one, 
adjusting the right number of epochs and hyperparameters really tested our understanding. Despite training for more than a 
day, we were unable to complete the second task due to the lack of a GPU. Still, we enjoyed the process.

Train.py - We used tensorflow to create a custom deep learning model using complicated CNN architecture. Some of the layers 
we used include - Conv2D, Dropout, MaxPool2D, Flatten, Dense. We also used ImageDataGenerator and coupled it with 
flow_from_dataframe to iterate over and label the training dataset. We successfully trained a model with over 99% training
accuracy and saved it as an .h5 file. 

inference1.py - We loaded the previously saved model and used it to evaluate on the specified test folder. As we will be 
provided with the relative path, we are assuming that the test folder will be created within this submission folder. Our
code is also creating the .csv file as expected.

Some points to pay attention to:
1. In the train.py file, we have used the absolute path of image data and annotations.csv files. Please change it to your
 computer's absolute path incase you wish to test the train.
2. Please do include our submission even if a technical glitch is there. We have tried our tried our best effort to make 
it run
3. Also, please consider us for the prizes ;))

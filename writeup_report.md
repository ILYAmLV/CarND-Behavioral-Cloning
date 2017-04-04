
# Self-Driving Car Engineer Nanodegree

## Behavioral Cloning - Project 3

---



[//]: # (Image References)

[image2]: ./examples/center.jpg "Center"
[image3]: ./examples/recovery1.jpg  "Recovery Image"
[image4]: ./examples/recovery2.jpg  "Recovery Image"
[image5]: ./examples/recovery3.jpg  "Recovery Image"
[image6]: ./examples/normal.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"


### My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results


The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

My model is inspired by Nvidia's [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) report and the submission by a fellow GitHubber [Justin Heaton](https://github.com/JustinHeaton/Behavioral-Cloning)

This project is an image classification problem. I used CNN because they have much fewer connections and parameters and so they are easier to train. 

| Layer         | Kernel size   | # of Kernel  | Stride  |
| :-----------: |:-------------:| :----------: | :------:|
| 1              | 3 x 3         |     16    |   2 x 2     |
| 2             | 3 x 3          |     24    |      1 x 2  |
| 3              | 3 x 3         |     36    |          |
|  4             |  2 x 2        |   48      |          |
| 5              |    2 x 2      |     48    |         ||    

I used a dropout layer after the first fully connected layer which helps to prevent overfitting to the training data.

The model was compiled with an adam optimizer (learning rate = 1e-04) and mean squared error for loss metric.

I set the number of epochs to 100 epochs, the model stopped training at epoch 42 because the validation loss stopped improving.

Here is the model summary:


```python
from keras.models import load_model
model = load_model('model.h5')
model.summary()
```

    Using TensorFlow backend.
    

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    batchnormalization_1 (BatchNorma (None, 20, 64, 3)     80          batchnormalization_input_1[0][0] 
    ____________________________________________________________________________________________________
    convolution2d_1 (Convolution2D)  (None, 9, 31, 16)     448         batchnormalization_1[0][0]       
    ____________________________________________________________________________________________________
    convolution2d_2 (Convolution2D)  (None, 7, 15, 24)     3480        convolution2d_1[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_3 (Convolution2D)  (None, 5, 13, 36)     7812        convolution2d_2[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_4 (Convolution2D)  (None, 4, 12, 48)     6960        convolution2d_3[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_5 (Convolution2D)  (None, 3, 11, 48)     9264        convolution2d_4[0][0]            
    ____________________________________________________________________________________________________
    flatten_1 (Flatten)              (None, 1584)          0           convolution2d_5[0][0]            
    ____________________________________________________________________________________________________
    dense_1 (Dense)                  (None, 512)           811520      flatten_1[0][0]                  
    ____________________________________________________________________________________________________
    dropout_1 (Dropout)              (None, 512)           0           dense_1[0][0]                    
    ____________________________________________________________________________________________________
    activation_1 (Activation)        (None, 512)           0           dropout_1[0][0]                  
    ____________________________________________________________________________________________________
    dense_2 (Dense)                  (None, 10)            5130        activation_1[0][0]               
    ____________________________________________________________________________________________________
    activation_2 (Activation)        (None, 10)            0           dense_2[0][0]                    
    ____________________________________________________________________________________________________
    dense_3 (Dense)                  (None, 1)             11          activation_2[0][0]               
    ====================================================================================================
    Total params: 844,705
    Trainable params: 844,665
    Non-trainable params: 40
    ____________________________________________________________________________________________________
    

#### 2. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) report. I thought this model might be appropriate because it is dealing with a similar situation.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set and trained for 10 epochs with an early stopping callback which stops training when validation loss does not improve.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, specifically the hard left turn in front of the lake,  to improve the driving behavior in these cases, I modified the `drive.py` file to implement an [adoptive throttle](https://github.com/aditbiswas1/P3-behavioral-cloning/blob/master/drive.py)

```python
if (abs(float(speed)) < 22):
        throttle = 0.5
    else:
        if (abs(steering_angle) < 0.1): 
            throttle = 0.3
        elif (abs(steering_angle) < 0.5):
            throttle = -0.1
        else:
            throttle = -0.3
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)
```

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself whenever it drifted off the track. These images show what a recovery looks like starting from the right edge of the track working it's way towards the center :

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would double the amount of data and correct the left-turn bias. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had `42,468` number of data points. I then preprocessed this data by adjusting the contract using the `skimage.exposure adjust_gamma` function.

I then split the data set and put `5%` of the data into a validation set. 

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

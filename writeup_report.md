# **Behavioral Cloning** 

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Nvidia.png "Model Visualization"
[image2]: ./examples/center.jpg "Center Image"
[image3]: ./examples/cropped.jpg "Cropped Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/normal.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model/model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Starting from the bottom the network consists of a normalization layer using a Keras lambda layer (model.py  line 66). 
Followed by five convolutional layers with 5x5 or 3x3 filter and depths between 3 to 64 (model.py lines 67-71). 
The model includes RELU layers to introduce nonlinearity after each convolutional layer.
Followed by four fully connected layers.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 73). 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 80-82). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 79).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Overall I'm following the Nvidia model architecture.
My first step was crop the image. Not all of these pixels contain useful information, however. In the image above, the top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car. Those are actually more distractions to the model.
![alt text][image3]
Then normalized to -1~1 from 0–255.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set(4:1). I found that Nvidia model had a low mean squared error on both training set and validation set. But I still put a dropout layers in order to reduce overfitting (model.py lines 73).
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 64-77).
First, crop the image to 80x320 from 160x320(model.py  line 64)
Then a normalization layer using a Keras lambda layer (model.py  line 66). 
Followed by five convolutional layers with 5x5 or 3x3 filter and depths between 3 to 64 (model.py lines 67-71). 
The model includes RELU layers to introduce nonlinearity after each convolutional layer.
Followed by four fully connected layers.
Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:
![alt text][image2]
To augment the data set, I also flipped images and angles thinking that this would double my data set. For example, here is an image that has then been flipped:
![alt text][image6]
![alt text][image7]

After the collection process, I had 8036 number of data points. 
I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as a tip from other. I used an adam optimizer so that manually training the learning rate wasn't necessary.
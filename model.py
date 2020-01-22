import os
import csv
import math

data_path = './data/'
model_path = './model/'
samples = []
with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i, line in enumerate(reader):
        if i != 0: # The first line is title, not a picture path
            samples.append(line)
            
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2) # Save 20% for validation

import cv2
import numpy as np
import sklearn

# Generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = data_path + 'IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # Flip the image todouble the set
                image_flipped = np.fliplr(center_image)
                angle_flipped = -center_angle
                images.append(image_flipped)
                angles.append(angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 80, 320  # Trimmed image format
drop_rate = 0.5 # Dropping rate
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D

model = Sequential()
# set up cropping2D layer
# trim image to only see section with road
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu')) 
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(drop_rate))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), \
            epochs=5, verbose=1)
model.save(model_path + 'model.h5')
exit()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 20:58:29 2018

@author: astaroth
"""

# importing the keras ilbraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initialising the CNN
classifier = Sequential()

# ------------------First Convolution Layer------------------
# Convolution2D
# no. of filters=32 of shape=(3,3)
# input shape =(64,64); no. of channels=3
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation='relu'))

# Pooling
# poolsize = dimension of the maxpool layer to be created;
classifier.add(MaxPooling2D(pool_size=(2,2)))

# ------------------Second Convolution Layer------------------
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flattening
classifier.add(Flatten())

# Full Connection
# A number of 100 is a good choice; we choose 128 as its a power of 2
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(
        rescale=1./255)

training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')
test_set = test_datagen.flow_from_directory('test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         samples_per_epoch=8000,
                         nb_epoch=25,
                         validation_data=test_set,
                         nb_val_samples=2000)

# making new predictions

import numpy as np
from keras.preprocessing import image

test_image_1 = image.load_img('single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image_2 = image.load_img('single_prediction/cat_or_dog_2.jpg', target_size=(64,64))

test_image_1 =  image.img_to_array(test_image_1)
test_image_2 =  image.img_to_array(test_image_2)

# increasing the image to a 4-dimensionsional
test_image_1 = np.expand_dims(test_image_1, axis=0)
test_image_2 = np.expand_dims(test_image_2, axis=0)

result = []
result.append(classifier.predict(test_image_1))
result.append(classifier.predict(test_image_2))

print(training_set.class_indices) # shows the mapping of the values

prediction = []
for res in result:
    if res == 1:
        prediction.append('dog')
    else:
        prediction.append('cat')
   
print(prediction)
        

#!/usr/bin/env python
"""
Generate tensorflow binary classification model for images, as model.h5.
Assume images are screenshots, and don't bother with augmentation.
Train the images found in in two subdirectories of training_set and test_set.
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import math
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

batch_size = 16
train_len = 72
test_len = 14

classifier.add(Convolution2D(batch_size, 3, 3, input_shape = (48, 54, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(batch_size, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# Assume all originals are 1080p: 1920x1080 pixels, and scale by 40x horizontally, 20x vertically to 48x54
training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(48, 54),
        save_to_dir="tmp_resized_images",
        batch_size=batch_size,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(48, 54),
        batch_size=batch_size,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=math.ceil(train_len / batch_size),
        epochs=10,
        validation_data=test_set,
        validation_steps=math.ceil(test_len / batch_size))

classifier.save("model.h5")

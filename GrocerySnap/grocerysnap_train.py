# -*- coding: utf-8 -*-
"""GrocerySnap_Train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AhwTt1nZzV7uHl5R1aPVltqSYf2ceY9f
"""

!pip install opendatasets

import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import opendatasets as od

od.download("https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition")

training_set = tf.keras.utils.image_dataset_from_directory(
    '/content/fruit-and-vegetable-image-recognition/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

test_set = tf.keras.utils.image_dataset_from_directory(
    '/content/fruit-and-vegetable-image-recognition/validation',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=512,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=256,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=36,activation='softmax'))

cnn.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

training_history = cnn.fit(x=training_set,validation_data=test_set,epochs=50)
training_loss, training_accuracy = cnn.evaluate(training_set)
print('Training Accuracy:', training_accuracy)

cnn.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

training_history = cnn.fit(x=training_set,validation_data=test_set,epochs=30)
training_loss, training_accuracy = cnn.evaluate(training_set)
print('Training Accuracy:', training_accuracy)

cnn.save("trained_model.h5")
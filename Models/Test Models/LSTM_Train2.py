!pip install opendatasets
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import opendatasets as od

od.download("https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition")

train_dir = '/content/fruit-and-vegetable-image-recognition/train'
test_dir = '/content/fruit-and-vegetable-image-recognition/validation'

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(64, 64),
        class_mode='categorical')

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Flatten, Conv2D, MaxPooling2D

from keras.layers import Reshape

model = Sequential()
model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'), input_shape=(None, 64, 64, 3)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(32, activation='sigmoid'))

num_channels = 3
input_shape = (64, 64, 3)
width, height = 64, 64
input_dim = width * height * num_channels
model.add(Reshape((-1, *input_shape)))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=50, validation_data=validation_generator)

train_loss, train_accuracy = model.evaluate(train_generator)
print('Training Accuracy:', train_accuracy)

model.save("trained_model.h5")
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

training_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/GrocerySnap/Dataset/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=16,  # decrease batch size
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
    '/content/drive/MyDrive/GrocerySnap/Dataset/validation',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=16,  # decrease batch size
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
cnn.add(tf.keras.applications.MobileNetV2(
        input_shape=(64,64,3), 
        include_top=False,
        weights='imagenet'))
cnn.add(tf.keras.layers.GlobalAveragePooling2D())
cnn.add(tf.keras.layers.Dense(units=36, activation='softmax'))

cnn.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

training_history = cnn.fit(x=training_set,validation_data=test_set,epochs=10)  # decrease number of epochs
training_loss, training_accuracy = cnn.evaluate(training_set)
print('Training Accuracy:', training_accuracy)

#cnn.save("trained_model.h5")

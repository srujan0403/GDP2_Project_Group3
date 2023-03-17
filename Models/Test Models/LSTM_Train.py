!pip install opendatasets
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import opendatasets as od

od.download("https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition")


def preprocess_image(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.vgg16.preprocess_input(image)
    vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    features = vgg16(image)
    features = tf.reshape(features, (features.shape[0], -1))
  
    return features

batch_size = 32
image_size = (224, 224)

training_set = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/fruit-and-vegetable-image-recognition/train',
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
    smart_resize=True
)

test_set = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/fruit-and-vegetable-image-recognition/validation',
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
    smart_resize=True
)

def process_data(image, label):
    image = preprocess_image(image)
    return image, label

training_set = training_set.map(process_data)
test_set = test_set.map(process_data)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=256, input_shape=(None, 25088)))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=36, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(training_set, validation_data=test_set, epochs=50)
training_loss, training_accuracy = model.evaluate(training_set)
print('Training Accuracy:', training_accuracy)
model.save('trained_model.h5')
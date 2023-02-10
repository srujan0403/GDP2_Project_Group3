import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow import keras
from keras import datasets, layers, models
import keras.preprocessing.image

dataset_path = 'C:\\Users\\S546941\\PycharmProjects\\GrocerySnap\\GroceryDataset\\'

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

training_images = training_images[:60000]
training_labels = training_labels[:60000]
testing_images = testing_images[:12000]
testing_labels = testing_labels[:12000]

model = models.Sequential()
model.add(layers.Conv2D(64, (3,3), activation= 'relu', input_shape=(32,32,3)))
model.add(layers. MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers. Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.model')

# model = models.load_model('C:/Users/S546444/PycharmProjects/ClassifyImage/image_classifier.model')
#
# img = cv.imread('Plane.jpg')
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#
#
# plt.imshow(img, cmap=plt.cm.binary)
#
# prediction = model.predict(np.array([img]) / 255)
# index = np.argmax(prediction)
# print(f'Prediction is {class_names[index]}')

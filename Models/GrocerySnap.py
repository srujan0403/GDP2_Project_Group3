import numpy as np
import tensorflow as tf
import keras
import scipy

dataset_path = 'C:\\Users\\s546941\\PycharmProjects\\Grocery\\GroceryDataset\\Training'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(dataset_path,
                                                    target_size=(100, 100),
                                                    batch_size=32,
                                                    class_mode='categorical')


num_classes = len(train_generator.class_indices)
classes = train_generator.class_indices

model = keras.Sequential([
    keras.layers.Conv2D(64, (4,4), activation='relu', input_shape=(100, 100, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (4,4), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(256, (4,4), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(num_classes, activation='sigmoid')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit_generator(train_generator,
                              epochs=30,
                              steps_per_epoch=200)
train_loss, train_accuracy = model.evaluate_generator(train_generator, steps=len(train_generator))
print(f'Training loss: {train_loss:.4f}, Training accuracy: {train_accuracy:.4f}')

model.save('fruits_and_vegetables_classifier.h5')


test_image = tf.keras.preprocessing.image.load_img('Ava.jpg', target_size=(100, 100))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)


prediction = model.predict(test_image)
prediction_class = np.argmax(prediction, axis=1)
prediction_label = [key for key, value in classes.items() if value == prediction_class[0]]
print(f'The predicted class is: {prediction_label[0]}')

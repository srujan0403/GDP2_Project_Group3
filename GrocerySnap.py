





model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit_generator(train_generator,
                              epochs=10,
                              steps_per_epoch=10)


model.save('fruits_and_vegetables_classifier.h5')

test_image = tf.keras.preprocessing.image.load_img('Potato.jpg', target_size=(150, 150))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)


prediction = model.predict(test_image)
prediction_class = np.argmax(prediction, axis=1)
prediction_label = [key for key, value in classes.items() if value == prediction_class[0]]
print(f'The predicted class is: {prediction_label[0]}')

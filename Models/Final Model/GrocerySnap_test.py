!pip install opendatasets

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import opendatasets as od


test_set = tf.keras.utils.image_dataset_from_directory(
    '/content/fruit-and-vegetable-image-recognition/test',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=False,
)

cnn = tf.keras.models.load_model('/content/drive/MyDrive/trained_model.h5')


loss, accuracy = cnn.evaluate(test_set)

print(f"Test accuracy: {accuracy:.2%}")

image = tf.keras.preprocessing.image.load_img(image_path,target_size=(64,64))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  
predictions = cnn.predict(input_arr)

print(predictions)

result_index = np.where(predictions[0] == max(predictions[0]))
print(result_index)

plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()

print("It's a {}".format(test_set.class_names[result_index[0][0]]))

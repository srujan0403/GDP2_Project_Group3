!pip install opendatasets
!pip install opendatasets
!pip install firebase
import opendatasets as od
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import opendatasets as od
import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

od.download("https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition")

cred = credentials.Certificate("/content/drive/MyDrive/Firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://grocery-inventory-cb542-default-rtdb.firebaseio.com/'
})

app = firebase_admin.get_app()


ref.child('apple').set({
    'name': 'apple',
    'quantity': 10
})

ref.child('banana').set({
    'name': 'banana',
    'quantity': 25
})

cnn = tf.keras.models.load_model('/content/drive/MyDrive/trained_model.h5')


image_path = '/content/fruit-and-vegetable-image-recognition/test/banana/Image_1.jpg'


img = cv2.imread(image_path)


plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()


ref = db.reference()
result_index = np.where(predictions[0] == max(predictions[0]))
fruit_veg_name = test_set.class_names[result_index[0][0]]
quantity = ref.child(fruit_veg_name).child('quantity').get()


print("It's a {}".format(fruit_veg_name))
print(f"Number of {fruit_veg_name}'s: {quantity}")


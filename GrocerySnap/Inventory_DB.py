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

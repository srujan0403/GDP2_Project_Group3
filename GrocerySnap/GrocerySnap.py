!pip install flask
!pip install pyngrok
!pip show firebase-admin
!pip install flask-ngrok


import os
import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import tensorflow as tf
from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok

cnn = tf.keras.models.load_model('/content/drive/MyDrive/GrocerySnap/Final/trained_model.h5')

app = Flask(__name__, template_folder='/content/drive/MyDrive/GrocerySnap/Final/templates')

run_with_ngrok(app)  

def classify_image(file_path):
   
    image = tf.keras.preprocessing.image.load_img(file_path, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

  
    test_set = tf.keras.utils.image_dataset_from_directory(
        '/content/drive/MyDrive/GrocerySnap/Dataset/test',
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(64, 64),
        shuffle=False,
    )
    predictions = cnn.predict(input_arr)
    result_index = np.where(predictions[0] == max(predictions[0]))
    fruit_veg_name = test_set.class_names[result_index[0][0]]

    
    try:
        cred = credentials.Certificate("/content/drive/MyDrive/GrocerySnap/Final/Firebase_key.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://grocery-inventory-cb542-default-rtdb.firebaseio.com/'
        }, name='grocery_app')
        app = firebase_admin.get_app()
    except ValueError as e:
        print("Firebase app already initialized:", e)
        app = firebase_admin.get_app(name='grocery_app')
    ref = db.reference(app=app) 

    quantity = ref.child(fruit_veg_name).child('quantity').get()
    return fruit_veg_name, quantity

def update_quantity(fruit_veg_name, new_quantity):
    
    try:
        cred = credentials.Certificate("/content/drive/MyDrive/GrocerySnap/Final/Firebase_key.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://grocery-inventory-cb542-default-rtdb.firebaseio.com/'
        }, name='grocery_app')
        app = firebase_admin.get_app()
    except ValueError as e:
        print("Firebase app already initialized:", e)
        app = firebase_admin.get_app(name='grocery_app')
    ref = db.reference(app=app) 
    ref.child(fruit_veg_name).update({'quantity': new_quantity})

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        
        file = request.files['file']

        
        file_path = os.path.join('./uploads', file.filename)
        file.save(file_path)

        
        fruit_veg_name, quantity = classify_image(file_path)

        
        return render_template('result.html', fruit_veg_name=fruit_veg_name, quantity=quantity)     

  @app.route('/update', methods=['POST'])
def update():
    if request.method == 'POST':
        
        fruit_veg_name = request.form['fruit_veg_name']
        new_quantity = request.form['new_quantity']

        
        update_quantity(fruit_veg_name, new_quantity)

        
        success_msg = f"Successfully updated the quantity of {fruit_veg_name} to {new_quantity}."

        
        return redirect(url_for('index', success_msg=success_msg)) 
   

if __name__ == '__main__':
    
    os.makedirs('./uploads', exist_ok=True)

    !ngrok authtoken 2Nd87EdFMonyCJ7lBTRetwoBo0i_3Ff7ii2U2tm6JKAkGu56T

    app.run()

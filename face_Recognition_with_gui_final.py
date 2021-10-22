# importing libraries

import tkinter as tk

from tkinter import Message, Text

import cv2

import os

import shutil

import csv

import numpy as np

from PIL import Image, ImageTk

import pandas as pd

import datetime

import time

import tkinter.ttk as ttk

import tkinter.font as font

from pathlib import Path
global pass1
global txt2 

window = tk.Tk() 

window.title("Face_Recogniser")

window.configure(background ='white')

window.grid_rowconfigure(0, weight = 1)

window.grid_columnconfigure(0, weight = 1)

message = tk.Label(

    window, text ="Face-Recognition-System", 

    bg ="green", fg = "white", width = 50, 

    height = 3, font = ('times', 30, 'bold')) 

     

message.place(x = 200, y = 20)
 


lbl2 = tk.Label(window, text ="path",

width = 20, fg ="green", bg ="white", 

height = 2, font =('times', 15, ' bold ')) 

lbl2.place(x = 400, y = 200)
 

txt2 = tk.Entry(window, width = 20, 

bg ="white", fg ="green", 

font = ('times', 15, ' bold ')  )

txt2.place(x = 700, y = 200)


lbl1 = tk.Label(window, text ="Key",

width = 20, fg ="green", bg ="white", 

height = 2, font =('times', 15, ' bold ')) 

lbl1.place(x = 400, y = 300)
 

txt1 = tk.Entry(window, width = 20,show='*', 

bg ="white", fg ="green", 

font = ('times', 15, ' bold ')  )

txt1.place(x = 700, y = 300)

def TakeImages(): 


   
    name=txt2.get()
    key=txt1.get()
    if(len(name)==0 or len(key)==0):
        res="Enter path and name to proceed"
        message.configure(text = res)
        return

    # Checking if the ID is numeric and name is Alphabetical

    if(len(name)>0): 
        
        import cv2
        import numpy as np
        
        # Load HAAR face classifier
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Load functions
        def face_extractor(img):
            # Function detects faces and returns the cropped face
            # If no face detected, it returns the input image
            
            #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(img, 1.3, 5)
            
            if faces is ():
                return None
            
            # Crop all faces found
            for (x,y,w,h) in faces:
                x=x-10
                y=y-10
                cropped_face = img[y:y+h+50, x:x+w+50]
        
            return cropped_face
        
        # Initialize Webcam
        cap = cv2.VideoCapture(0)
        count = 0
        
        # Collect 100 samples of your face from webcam input
        while True:
        
            ret, frame = cap.read()
            if face_extractor(frame) is not None:
                count += 1
                face = cv2.resize(face_extractor(frame), (400, 400))
                #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
                # Save file in specified directory with unique name
                file_name_path = name+'/' + str(count) + '.jpg'
                cv2.imwrite(file_name_path, face)
        
                # Put count on images and display live count
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Cropper', face)
                
            else:
                print("Face not found")
                pass
        
            if cv2.waitKey(1) == 1 or count == 1000: #1 is the Enter Key
                break
        cap.release()
        cv2.destroyAllWindows()      
        print("Collecting Samples Complete")
        fencrypt(name,key)
             
# Training the images saved in training image folder    

def TrainImages():

    name=txt2.get()
    key=txt1.get()
    if(len(name)==0 or len(key)==0):
        res="Enter path of dataset and name to proceed"
        message.configure(text = res)
        return
    fdecrypt(name,key)
    os.chdir('C:/Users/Noman/Desktop/ifscrtrial1/Deep-Learning-Face-Recognition-master/')
    from keras.layers import Input, Lambda, Dense, Flatten
    from keras.models import Model
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input
    from keras.preprocessing import image
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    import numpy as np
    from glob import glob
    import matplotlib.pyplot as plt

    IMAGE_SIZE = [224, 224]

    train_path = 'dataset/train'
    valid_path = 'dataset/test'

    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    for layer in vgg.layers:
      layer.trainable = False


    folders = glob('dataset/train/*')

    x = Flatten()(vgg.output)
    # x = Dense(1000, activation='relu')(x)
    prediction = Dense(len(folders), activation='softmax')(x)

    model = Model(inputs=vgg.input, outputs=prediction)

    model.summary()

    model.compile(
      loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
    )


    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('dataset/train',
                                                     target_size = (224, 224),
                                                     batch_size = 34,
                                                     class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory('dataset/test',
                                                target_size = (224, 224),
                                                batch_size = 34,
                                                class_mode = 'categorical')


    r = model.fit_generator(
      training_set,
      validation_data=test_set,
      epochs=20,
      steps_per_epoch=len(training_set),
      validation_steps=len(test_set)
    )

    import tensorflow as tf

    from keras.models import load_model

    model.save('facefeatures_new_model.h5')

    fencrypt(path,key)
    
def TrackImages():

    from PIL import Image
    from tensorflow.keras.applications.vgg16 import preprocess_input
    import base64
    from io import BytesIO
    import json
    import random
    import cv2
    from tensorflow.keras.models import load_model
    import numpy as np
    
    from tensorflow.keras.preprocessing import image
    os.chdir('C:/Users/Noman/Desktop/ifscrtrial1/Deep-Learning-Face-Recognition-master/')
    model = load_model('facefeatures_new_modelx2.h5')
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    def face_extractor(img):
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        
        if faces is ():
            return None
    
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            cropped_face = img[y:y+h, x:x+w]
    
        return cropped_face
    
    video_capture = cv2.VideoCapture(0)
    while True:
        _, frame = video_capture.read()
        #canvas = detect(gray, frame)
        #image, face =face_detector(frame)
        
        face=face_extractor(frame)
        if type(face) is np.ndarray:
            face = cv2.resize(face, (224, 224))
            im = Image.fromarray(face, 'RGB')
               #Resizing into 128x128 because we trained the model with this image size.
            img_array = np.array(im)
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
            print(pred)
                         
            name="None matching"
            
            if(pred[0][1]>0.5):
                name='numan'
            cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
def fencrypt(path,key):
    import os
    # try block to handle exception

    try:

        # take path of image as a input

        #path="C:/Users/Noman/Desktop/ifscrtrial1/Deep-Learning-Face-Recognition-master/elon"
        # taking encryption key as input

        #key = input('Enter Key for encryption of Image :')
        # open file for reading purpose
        os.chdir(path)
        def encrypt(path):
            os.chdir(path)
            for i in os.listdir():
                if('.jpg' in i):
                    f=open(i,'rb')
                    image = f.read()
                    f.close()
                    image = bytearray(image)
                    for x in key:
                        for index, values in enumerate(image):
                            image[index] = ~(~values ^ ord(x))
                    f=open(i,'wb')
                    f.write(image)
                    f.close()
                else:
                    encrypt(path+"/"+i)
        for i in os.listdir():
            if('.jpg' in i):
                encrypt(path)
            else:
                encrypt(path+"/"+i)            
    except Exception as e:
        print('Error caught : ', e)
        return
def fdecrypt(path,key):
    
    import os
    # try block to handle exception

    try:

        # take path of image as a input

        #path="C:/Users/Noman/Desktop/ifscrtrial1/Deep-Learning-Face-Recognition-master/elon"
        # taking encryption key as input

        #key = input('Enter Key for decryption of Image :')
        # open file for reading purpose
        os.chdir(path)
        def decrypt(path):
            os.chdir(path)
            for i in os.listdir():
                if('.jpg' in i):
                    f=open(i,'rb')
                    image = f.read()
                    f.close()
                    image = bytearray(image)
                    for x in key:
                        for index, values in enumerate(image):
                            image[index] = ~(~values ^ ord(x)) 
                    f=open(i,'wb')
                    f.write(image)
                    f.close()
                else:
                    decrypt(path+"/"+i)
        for i in os.listdir():
            if('.jpg' in i):
                decrypt(path)
            else:
                decrypt(path+"/"+i)            
    except Exception as e:
        print('Error caught : ', e)
takeImg = tk.Button(window, text ="Sample", 

command = TakeImages, fg ="white", bg ="green", 

width = 20, height = 3, activebackground = "Red", 

font =('times', 15, ' bold '))

takeImg.place(x = 200, y = 500)

trainImg = tk.Button(window, text ="Training", 

command = TrainImages, fg ="white", bg ="green", 

width = 20, height = 3, activebackground = "Red", 

font =('times', 15, ' bold '))

trainImg.place(x = 500, y = 500)

trackImg = tk.Button(window, text ="Testing", 

command = TrackImages, fg ="white", bg ="green", 

width = 20, height = 3, activebackground = "Red", 

font =('times', 15, ' bold '))

trackImg.place(x = 800, y = 500)

quitWindow = tk.Button(window, text ="Quit", 

command = window.destroy, fg ="white", bg ="green", 

width = 20, height = 3, activebackground = "Red", 

font =('times', 15, ' bold '))

quitWindow.place(x = 1100, y = 500)
 

  
window.mainloop()
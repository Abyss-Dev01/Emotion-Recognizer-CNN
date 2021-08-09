#importing necessary libraries
import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

model = model_from_json(open("fer.json","r").read())
model.load_weights('fer.h5')
#face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

test_image = cv2.imread('../expression/disgust2.jpg')
gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


faces = face_cascade.detectMultiScale(gray_image, 1.1,4)
#Drawing rectangle around the face
for(x,y,w,h) in faces:
    cv2.rectangle(test_image,(x,y),(x+w,y+h),(255,0,0))
    roi_gray = gray_image[y:y+w,x:x+h]
    roi_gray = cv2.resize(roi_gray,(48,48))
    image_pixels = img_to_array(roi_gray)
    image_pixels = np.expand_dims(image_pixels,axis = 0)
    image_pixels /= 255
    #emotion prediction on the image
    predictions = model.predict(image_pixels)
    max_index = np.argmax(predictions[0])
    emotion_detection = ('angry','disgust','fear','happy','sad','surprise','neutral')
    emotion_prediction = emotion_detection[max_index]
    print(emotion_prediction)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50,50)
    fontScale = 1
    color = (255,0,0)
    thickness = 2
    image = cv2.putText(test_image, emotion_prediction, org, font, fontScale, color, thickness, cv2.LINE_AA)
#cv2.resize(test_image,(48,48))
cv2.imshow("Emotion Prediction",cv2.resize(test_image,(500,400)))
cv2.waitKey()

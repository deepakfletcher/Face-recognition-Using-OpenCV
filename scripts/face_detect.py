import numpy as np
import cv2
import pickle
import os
from PIL import Image

face_detect = cv2.CascadeClassifier('./CV2_cascades_folder/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load("./TrainData/tain_data.yml")

labels = {"person_name": 1}
with open("./Pickle/lbls.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

capture = cv2.VideoCapture(0)

while(True):
    ret, frame = capture.read()
    gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
    	roi_gray = gray[y:y+h, x:x+w]
    	roi_color = frame[y:y+h, x:x+w]


    	id_, confidence = recognizer.predict(roi_gray)
    	if confidence>=40 and confidence <= 85:
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		cv2.putText(frame,labels[id_],(x,y+h),font,1,(255, 0, 0),2, cv2.LINE_AA)

    	cv2.rectangle(frame,(x, y),((x+w),(y+h)),(225,0,0),3)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

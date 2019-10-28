import os
import cv2
from PIL import Image
import numpy as npy
import pickle

currentid=0
labelids={}
xtrain=[]
ylabels=[]

basedir=os.path.dirname(os.path.abspath(__file__))
imagesdir=os.path.join(basedir,"Face_Datas")

face_detect= cv2.CascadeClassifier('./CV2_cascades_folder/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.createLBPHFaceRecognizer()

for root,dirs,files in os.walk(imagesdir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path=os.path.join(root,file)
            label=os.path.basename(root).replace(' ','-').lower()
            print(label, path)

            if not label in labelids:
                labelids[label]=currentid
                currentid+=1
            id_=labelids[label]
            print(labelids)

            pilimage=Image.open(path).convert("L")
            image_array=npy.array(pilimage,"uint8")
            print(image_array)
            faces=face_detect.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            for x,y,w,h in faces:
                regionofintrest = image_array[y : y+h , x : x+w]
                xtrain.append(regionofintrest)
                ylabels.append(id_)



with open("./Pickle/lbls.pickle","wb") as f:
    pickle.dump(labelids,f)

recognizer.train(xtrain,npy.array(ylabels))
recognizer.save("./TrainData/tain_data.yml")
print("face Training finished .........")

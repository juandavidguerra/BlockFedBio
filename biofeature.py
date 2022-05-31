
## https://ritik12.medium.com/facial-recognition-using-pytorch-and-opencv-467c4e41d1f
## https://towardsdatascience.com/face-landmarks-detection-with-pytorch-4b4852f5e9c4

import fnmatch
import os
from matplotlib import pyplot as plt
import cv2


face_cascade = cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')# Load the cascade

paths="/data/"


for root,_,files in os.walk(paths):
    for filename in files: 
        file = os.path.join(root,filename)
        if fnmatch.fnmatch(file,'*.jpg'):
            
            img = cv2.imread(file)        
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # Draw rectangle around the faces
            for (x, y, w, h) in faces:
              crop_face = img[y:y+h, x:x+w]
            path = os.path.join(root,filename)
            cv2.imwrite(path,crop_face)
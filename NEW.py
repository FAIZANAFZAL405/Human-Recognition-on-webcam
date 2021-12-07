import cv2
import numpy as np
import face_recognition
# import image

from pathlib import Path
from PIL import Image

path = "images"


 # loop and import the images path
for path in Path("images").glob("*.jpg"):

    knownImg = face_recognition.load_image_file(path)
    knownImgEncode = face_recognition.face_encodings(knownImg)




# convert image colour
knownImg = cv2.cvtColor(knownImg,cv2.COLOR_BGR2RGB)

# import test image

imgtest = face_recognition.load_image_file('arslantest.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

# find face in known image

faceLoc = face_recognition.face_locations(knownImg)[0]
knownImgEncode = face_recognition.face_encodings(knownImg)[0]
cv2.rectangle(knownImgEncode,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


# find face on test image
faceLoctest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,255),2)

# compare encoding of both images

results = face_recognition.compare_faces([knownImgEncode], encodetest)
# distance between two images
faceDis = face_recognition.face_distance([knownImgEncode], encodetest)

print(results, faceDis)
cv2.putText(imgtest,f'{results} {round(faceDis[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('known img',knownImg)
cv2.imshow('test image',imgtest)
cv2.waitKey(0)

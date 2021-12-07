import cv2
import numpy as np
import face_recognition
import os

path = 'DATA' images = []
classNames = []
myList = os.listdir(path)
print(myList)


# importing images
for cls in myList:
    currentimage = cv2.imread(f'{path}/{cls}')
    images.append(currentimage)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

# function for find encoding
def findEncodings(images):
    encodeList = []
    for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
    return encodeList

encodelistknown = findEncodings(images)
print('encoding finish')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
# resise image for fast process and convert into RGB
    imagesize = cv2.resize(img,(0,0),None,0.25,0.25)
    imagesize = cv2.cvtColor(imagesize, cv2.COLOR_BGR2RGB)
    # finding face in webcam
    facesinframe = face_recognition.face_locations(imagesize)
# encoding of current frame in webcam
    encurframe = face_recognition.face_encodings(imagesize,facesinframe)

    for encodeFace,faceloc in zip(encurframe,facesinframe):
        matches = face_recognition.compare_faces(encodelistknown,encodeFace)
        facedist = face_recognition.face_distance(encodelistknown,encodeFace)
        print(facedist)
        matchIndex = np.argmin(facedist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)



    cv2.imshow('webcam',img)
    cv2.waitKey(1)



















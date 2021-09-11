import cv2
import numpy as np
def face_detect(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Detecta las caras
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    # Dibuixa un rectangle a les cares
    cares=[]
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        i=img[y:y+w,x:x+h]
        i=cv2.resize(i, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        cares.append(i)

    # Mostra el resultat
    #cv2.imshow('img2',cares[0])
    #cv2.waitKey()


    return cares

#a=cv2.imread('provam.jpg')
#b=cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
#face_detect(a)

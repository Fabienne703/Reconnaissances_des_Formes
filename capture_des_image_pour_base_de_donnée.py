#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on mond Dec 27 13:20:03 2020

@author: fabienne
"""

"""
1- mettre en oeuvre une base de donnée anoté contenant l'id de l'image 
2- affecter les valeurs dans un tableau
3- ensuite ecrire le code permettant de detecter et affecter les nom au image cacpturer dans le films
"""

import cv2
import operator
import common as c

face_cascade=cv2.CascadeClassifier('fichier_xml/haarcascade_frontalface_alt2.xml')

profile_cascade=cv2.CascadeClassifier('fichier_xml/haarcascade_profileface.xml')


cap=cv2.VideoCapture('extraire_cadres/activity_data/UCF-101/train/video/cleg.mp4')

id=0
while True:
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(5,5))
    for x, y, w, h in face:
        cv2.imwrite('data/photos/train_pic/cleg-{:d}.png'.format(id), frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        id+=1
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('a'):
        for cpt in range(3):
            ret, frame=cap.read()

    cv2.imshow('video', frame)

cap.release()
cv2.destroyAllWindows()
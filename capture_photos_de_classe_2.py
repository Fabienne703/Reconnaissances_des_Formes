#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on mond Dec 27 13:20:03 2020

@author: fabienne
"""

import cv2
import operator

#fichier contenant toutes les carractéristiques techniques de l'objet que nous recherchons dans nos image 
#elle sont creer a patir de fonction d'apprentissage inclus dans opencv
face_cascade=cv2.CascadeClassifier('data/fichier_xml/haarcascade_frontalface_alt2.xml')
profile_cascade=cv2.CascadeClassifier('data/fichier_xml/haarcascade_profileface.xml')

'''
se que dois faire notre algo est de scaner la photos de en bas, de droite a gauche afin de rechercher 
les objet decrite decrite dans le fichier XML.
ensuite il va, ensuite faire un changement d'echelle et va recommancer un certain nombre de fois et le coefficient 
correspondant a se changement de parametre est ce qu'on appelle le "scalefactor" "minneighbors permet d'eviter de repre
senter plusieur rectangle autour du visage detecter

en effet les multiple carrer constater dans les differente image correspondent au differente échelle qui 
apparaisse et change dans l'image ne meme temps
 du coup quand on change le scale factor on a beaucoup moins de caré et aussi pour eviter les faux positif on utilise 
 minneighbor il verifie que dans le nombre de couche a coter si l'objt est detecter 
 et l'algorihtme detecteras sur les trois couche si l'objet est bien la
'''

#lancement de notre camera
cap=cv2.VideoCapture('data/fichier_video/classe_1_classe/cleg.mp4')

#recuperation de la largeur de la webcam sa nous servira pour rogner les images
width=int(cap.get(3))
marge=70

id=0
while True:
    #frame: contient l'image
    ret, frame=cap.read()
    tab_face=[]
    
    #mesure de temps d'execution algo  de notre image avec getTickcount()
    tickmark=cv2.getTickCount()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #face contient une liste de quadruplet que nous allons récuperer
    #minSize : permet de suprimer les rectangle ayant une valeur innferieur a 5,5
    face=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(80, 80))
    #x,y: cordonnée; w,h: hauteur et largeur;
    for x, y, w, h in face:
        cv2.imwrite('data/photos/train_pic/cleg/p-{:d}.png'.format(id), frame[y:y+h, x:x+w])
        #cv2.rectangle(frame,(x,y),(x+w, y+h), (255, 0, 0), 2)
        tab_face.append([x, y, x+w, y+h])
        id+=1
    face=profile_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
    
    for x, y, w, h in face:
        cv2.imwrite('data/photos/train_pic/cleg/p-{:d}.png'.format(id), frame[y:y+h, x:x+w])
        tab_face.append([x, y, x+w, y+h])
        id+=1
    gray2=cv2.flip(gray, 1)
    face=profile_cascade.detectMultiScale(gray2, scaleFactor=1.2, minNeighbors=4)
    
    for x, y, w, h in face:
        cv2.imwrite('data/photos/train/cleg/p-{:d}.png'.format(id), frame[y:y+h, x:x+w])
        tab_face.append([width-x, y, width-(x+w), y+h])
        id+=1
    tab_face=sorted(tab_face, key=operator.itemgetter(0, 1))
    index=0
   
    for x, y, x2, y2 in tab_face:
        if not index or (x-tab_face[index-1][0]>marge or y-tab_face[index-1][1]>marge):
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
        index+=1
        
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('a'):
        for cpt in range(3):
            ret, frame=cap.read()
    
    #affichage du temps d'execution
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    
    #affichage du text dans la video
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow('video', frame)

    for cpt in range(1):
        ret, frame = cap.read()
        
cap.release()
cv2.destroyAllWindows()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CrCreated on mond Dec 27 13:20:03 2020

@author: fabienne
"""

#Partie 1 -

"""la premiere etape est d'importer les librairie qui nous aiderons dans
l'importation et le traitements sur nos differentes images."""
""" initialisation ANNs"""

from tensorflow.keras.models import Sequential
"""operation de convolution"""
from tensorflow.keras.layers import Convolution2D
"""Pooling reduction image""" 
from tensorflow.keras.layers import MaxPooling2D
"""flattenign pour applatir pour entrer ANN""" 
from tensorflow.keras.layers import Flatten
""" pour ajouter des couche cachée et connecter"""
from tensorflow.keras.layers import Dense
from keras.utils import Sequence

from tensorflow.keras.layers import Dropout

# initialisation du CNN de neurone a convolution comme les ANNs
classifier = Sequential()

# step 1: convolution ajout de la couche de convolution
"""
    - dans cette partie pour la creation de notre couche de convolution nous 
devons definir dans cette etape le nombre de feature detector que nous allons 
utiliser elle correspond en meme temps au nombre de features maps que nous allons
creer car pour chaque features detector correspond un features maps donné
    - filters= dimensionalité espace de sortie === nombre de feature detector c'est a dire de filtre
    comme remarque ici si nous avons une deuxieme couche de convolution, alors le nombre de filtre dois 
    doubler normalement c'est a dire 64 dans autre ccas ainsi de suite. cella tu peux expliquer
    
    -kernel_size= elle correspond a la taille de la matrice de notre filters
    sa pouvait etre de la forme [3, 3] ou [3,5...]
    
    -strides= taille de deplacement de pixel 1 ou 2 quand on effectue l'operation de convolution
   
    -inpur_shape= permet de definir la taille de nos image a lire(forcer les image a adopter le meme format) et le second 
    argument 3 permet de dire que nous manipulons des images couleurs RGB
    
    -activation= pour ajouter de la non lineariter dans le modele
    permet de remplacer toutes les valeurs négative par des 0.
    -relu correspond a la fonction redresseur comme fonction d'activation
"""
classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1,
                             input_shape=(200, 200, 3),
                             activation = "relu"))



# step 2: Pooling
"""
elle consiste a prendre la feauture maps que nous avons obtenue juste avant 
l'etape de convolution et on va prendre les case 2/2 on construit comme sa jusqu'aobtenir 
un plus petit resultat
    
- pool_size=permet de definir la taille de notre matrice de selection du maximun
"""
classifier.add(MaxPooling2D(pool_size=(2,2)))


# ajout de la nouvelle couche de convolution faut pas oublier son pooling
classifier.add(Convolution2D(filters=64, kernel_size=3, strides=1,
                             activation = "relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))



# a present pour melanger les deux couche de convolution on dois multiplier par 64 filtre a present 32*2

classifier.add(Convolution2D(filters=128, kernel_size=3, strides=1,
                             activation = "relu"))

classifier.add(MaxPooling2D(pool_size=(2,2)))



# step 3: Flattening
"""
    -phase d'aprantissage pour obtenir des input pour notre ANNs
    
elle se fait a la fin pour permettre de renseigner de bonne information au neurone
"""
classifier.add(Flatten())

# step 4: ANNs completement connecté
"""
    - Dense = permet d'ajouter une couche de neurone caché
        -units= nombre de neurone qui appartiennent a la couche
            dans le cas des reseaux de neurone artificielle 
            on a dis que nous pouvions prendre le nombre de variable ici nous ne poiuvons 
            pas definir normalement 
            alors dans notre cas on aura bcp de features faut prendre les nombre 
            puissance de 2 sa marche tres bien
        -activation= represente la fonction d'activation pour cette couche
        - relu est tres utiliser pour sa particularité d'etre stricte 
        soit elle laisse passer le signal ou non
"""
classifier.add(Dense(units=256, activation="relu"))
classifier.add(Dropout(rate=0.3))
classifier.add(Dense(units=256, activation="relu"))
classifier.add(Dropout(rate=0.3))
classifier.add(Dense(units=256, activation="relu"))
classifier.add(Dropout(rate=0.3))



# definiton de la couche de sortie de notre reseau de neurone a convolution
"""
 - pour la couche de sortir puisque 
 nous somme tjr dans le contexte de classification 
 alors nous utilisons la fonction sigmoid sinon on aurait utilise dans un cadre c
 catégorielle la fonction softmax et nous avons juste besoin de 1 neurone
"""
classifier.add(Dense(units=6, activation="softmax"))



# etape de compilation de notre reeseau de neurone.
"""
    - optimizer= correspond a l'algorithme de macine learning a utiliser pour la classification
        adam correspond au stochastique de merde
    -loss= represente la fonction de cout binary_cross.. pour la classification et categorical_cros... pour la regression
    -metrics= "accuracy" 
    
"""
classifier.compile(optimizer="adam",
                   loss="categorical_crossentropy",
                   metrics=['accuracy'])


#########################################################
# Entrainement de notre réseaux de neurone a convolution#
#########################################################

"""
    - faut aller lire dans la documentation de keras a keras documentation
    - augmentation d'image : permet d'eviter le surentrainement sur le jeux de donné il permet de 
    modifier le jeux de donnée de toutes les formes et de transformer les images et nous permettra d'avoir beaucoup plus 
    d'image differente
"""
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


training_set = train_datagen.flow_from_directory(
        'extraire_cadres/activity_data/test/train/images',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'extraire_cadres/activity_data/test/photo_test/images',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical')

"""
    pour obtenir le nombre de validation_steps, on divise le nombre de donnée
    du dataset par le nombre de batch_size::::: 2000/32....
    - pour le training_set  = on divise par le nombre d'observation de notre 
    training set par le nombre de batch_size se qui donne 8000/32=250
    - pour le validation_test= ici on effectue le meme processus pour le training
    set mais on prend par contre l'echantillon de test a cette fois 2000/32 = 62.5 ===63
    -nous avons mentionner lors de la construction des ANNs celle-la permet d'evaluer le reseau au fur et a mesure qu'on l'entraine
    pour ne pas l'evaluer a la fin de l'apprentissage en meme temps 
    ici on fait tout a la fois comme le k-cross... evaluation et ajustement de paramètre
"""

classifier.fit_generator(
        training_set,
        steps_per_epoch=82,
        epochs=10,
        validation_data=test_set,
        validation_steps=10)


#evaluation du modele
classifier.evaluate(test_datagen, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

#pour enregistrer le model apres entrainement

classifier.save('train_avec_cnn_accuracy_faible.h5')

"""pour ouvrir le fichier apres entrainement on utilise le classifier.load()"""


#  dANS CETTE NOUVELLE PHASE NOUS ALLONS PASSER A LA PRÉDICTION D'ANIMAUX CHIEN OU CHAT
"""
    - ici il ne s'agit pas de manipuler des matrice mais plutot des image alors nous
    devons les importers dans l'endroit ou il se trouve grace a des bibliotheque de keras 
    - ensuite penser a dimenssionner notre images a la taille voulu
    - et lancer la prediction comme avec les ANNs
"""



import numpy as np
from keras.preprocessing import image

# importation de notre image en spécifiant la taille qui correspond forcement a celle de l'entrainement

test_image = image.load_img('extraire_cadres/activity_data/test/train/images',
                            target_size=(200, 200))

# ajout d'une quatrieme dimenssion a notre image a l'indice 0 pour permettre l'evaluation par notre CNN
# axis permet de spécifier l'index du groupe 
# car nous avons dans notre cas le premier groupe si nous avons plusieurs groupe on peut les positionner de la meme facon

test_image = np.expand_dims(test_image, axis=0)

# transformation de notre image en array un tableaux d'element
# test_image = image.img_to_array(test_image)

# prediction sur notre image chargé

result = classifier.predict(test_image)

# maintenant il nous faut spécifier a quoi correspond chaque prédiction 0,1,...,6
training_set.class_indices

# on peut maintenant mettre le resultat dans une variable et afficher

if result[0][0]==1:
    prediction = "Je viens de trouver Fabienne"
elif result[0][1]==1:
    prediction = "Je viens de trouver astrel"
elif result[0][2]==1:
    prediction = "Je viens de trouver Leon"
elif result[0][3]==1:
    prediction = "Je viens de trouver Michelet"
elif result[0][4]==1:
    prediction = "Je viens de trouver Peterson"
else:
    prediction = "Je viens de trouver bernard"
    
    
    
"""
POUR AMELIORER UN MODÈLE ON PEUT :
        - Changer la taille de l'image
        - ajouter plusieurs couche de convolution
        - ajouter de nouvelle couche de reseaux de neurone et pour eviter 
        de tomber dans les cas de surapprentissage alors ajouter le drop-out 
        pour  definir le taux d'apprentissage qui permet de ne pas construire un réseaux de neurone
        qui apprend trop il permet de désactiver les neurones qui apprenent trop.
        - tous ses éléments permettent d'améliorer les performance de notre modèle
        et eviter le surapprentissage lorsque le taux d'apprentissage sur de nouvelle 
        donnée est tres inférieur a celle de donnée d'entrainement.
"""
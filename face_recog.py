# l'imprtation de librairies necessaires
# python3 face_recog.py
#import face_recognition
import face_recognition as fr
import cv2 as cv
from datetime import datetime

# from PIL import Image

# ici on fait l'importation de l'enemble images des personnes faisant parti de la
# base de donnees images de notre model forme
global face
global face_locations
global frame
img = fr.load_image_file("images/Peterson/1.png")
img1 = fr.load_image_file("images/Astrel/1.png")
img2 = fr.load_image_file("images/Fabienne/1.png")
img3 = fr.load_image_file("images/Bernard/1.png")

# img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# img1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
# img2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
# img3=cv.cvtColor(img3,cv.COLOR_BGR2GRAY)

img_enc = fr.face_encodings(img)[0]
img1_enc = fr.face_encodings(img1)[0]
img2_enc = fr.face_encodings(img2)[0]
img3_enc = fr.face_encodings(img3)[0]
known_face_encodings = [
    img_enc, img1_enc, img2_enc, img3_enc
]
known_face_names = [
    "Peterson", "Astrel", "Fabienne", "Bernard"
]

# dans cette partie on a defini la fonction qui en compte l'heure que la
# personne s'est presente dans l'espace designe
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        # if name not in nameList:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.writelines(f'\n{name},{dtString}')


# on a definit le lancement de la video pour la reconnaissance des personnes
cap = cv.VideoCapture(0)


# cv.CascadeClassifier()
def face_identify():
    try:
        for (top, right, bottom, left), face_encoding in zip(face_locations, frame_encs):
            matches = fr.compare_faces(known_face_encodings, face_encoding)
            name = "Inconnu"
            faceDis = fr.face_distance(known_face_encodings, face_encoding)
            print(faceDis)
            # fr  = face_locations.face_distance(known_face_encodings,face_encoding)
            # match_index= np.argmin(match_index)
            if True in matches : #and matches.index(True):
                match_index = matches.index(True)
                name = known_face_names[match_index]
                #print(known_face_names)
            cv.putText(frame, name, (left, top), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            markAttendance(name)
    except:
        pass

while (True):
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    try:
        frame_encs = fr.face_encodings(frame, face_locations)
    except:
        print("prob in gray frame")
    face_locations = fr.face_locations(gray)

    # print(face_locations)
    face_identify()

    cv.imshow("Camera", frame)

    # print(len(face_locations))
    key = cv.waitKey(1)
    if (key == 27): break

cap.release()
cv.destroyAllWindows()

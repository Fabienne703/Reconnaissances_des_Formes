import cv2 as cv

face_cascade=cv.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
cap=cv.VideoCapture(0)
while(1):
    _,frame=cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.5,5)
    for  (x,y,a,b) in faces:
        #print(top,right,bottom,left)
        cv.rectangle(frame,(x,y),(x+a,y+b),(0,0,255),2)


    cv.imshow("Camera",frame)
    key=cv.waitKey(1)
    if key==27:
        break
cap.release()
cv.destroyAllWindows()
import cv2
import numpy as np

recog = cv2.face.LBPHFaceRecognizer_create()
recog.read("jackData_trainner.yml")
face_xml = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

names = ["name1", "name2"]


cap = cv2.VideoCapture(0)
isOpened = cap.isOpened()

while(isOpened):
    (flag, frame) = cap.read()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_xml.detectMultiScale(frameGray, 1.4, 3)

    for(x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 1), 2, cv2.LINE_AA)
        ids, confidence = recog.predict(frameGray[y:y+h, x:x+w])
        print(confidence)
        if confidence > 70:
            cv2.putText(frame, "Who are you?", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(frame, names[ids-1], (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("test", frame)
    if ord("q") == cv2.waitKey(1):
        break

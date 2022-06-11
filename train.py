import cv2
import numpy as np

face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def samples_and_labels():
    faceData = []
    ids = []
    for i in range(1, 2000):
        path = "../03_DataSet/01_Grocery/image" + str(i) + ".jpg"
        img = cv2.imread(path)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_xml.detectMultiScale(imgGray)

        for (x, y, w, h) in faces:
            if i <= 1000:
                ids.append(1)
            else:
                ids.append(2)
            faceData.append(imgGray[y:y+h, x:x+w])

    return faceData, ids


(faces, ids) = samples_and_labels()
print(faces, ids)
print("Training...")
jackData = cv2.face.LBPHFaceRecognizer_create()  # 创建LBPH

jackData.train(faces, np.array(ids))  # 参数1 为人脸像素数据  参数2 为对应人脸标签
jackData.save("jackData_trainner.yml")
print("Finish")

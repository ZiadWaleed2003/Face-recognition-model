import cv2
from Face_Model import FaceModel
import os

# change that var to the path of your image folder
img_path = r"C:\Users\ziad\PycharmProjects\Face Recognition model\test"

os.chdir(img_path)

model = FaceModel()

model.load_images(img_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    face_location, face_name = model.detect_faces(frame)

    for face_loc, name in zip(face_location, face_name):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

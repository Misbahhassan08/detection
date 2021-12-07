from __future__ import print_function
import requests
import json
import cv2
import sqlite3

class Detection:
    def __init__(self):
        self.cascadePath = "haarcascade_frontalface_alt2.xml"
        self.cascadePath_eye = "haarcascade_eye.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)
        self.eyeCascade = cv2.CascadeClassifier(self.cascadePath_eye)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.facing = 0
        self.total_faces = 0
        self.message = ""

    def detect_face(self, num):
        streaming = cv2.VideoCapture(0)

        while True:
            try:
                crop = []
                _label = []
                _, im = streaming.read()
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(gray)
                self.facing = 0
                self.message= ""
                self.total_faces = 0
                if len(faces) > 0:
                    self.total_faces = len(faces)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 2)
                        cv2.putText(im, str(len(faces)), (x,y-40), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2, 2)
                        crop.append(im[y:y + h, x:x + w])
                    for img in crop:
                        tracking = self.eyeCascade.detectMultiScale(img)
                        if len(tracking) > 0:
                            self.facing = self.facing + 1
                        pass
                cv2.imshow("Frame", im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                self.message = 'Faces = {}, Tracked : {}'.format(self.total_faces, self.facing)
                print(self.message)
            

            except Exception as error:
                print('EEEEEEEEEEEEEEEEEEEEEeroor : {}'.format(error))

        streaming.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detect = Detection()
    detect.detect_face(0)

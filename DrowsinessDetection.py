# Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
# face_utils for basic operations of conversion
from imutils import face_utils
from gtts import gTTS
import os
import time


def alarm1():
    os.system("start wakeup.mp3")
    time.sleep(2)

def alarm2():
    os.system("start freshair.mp3")
    time.sleep(2)

# Initializing the camera and taking the instance
vid = cv2.VideoCapture(0)

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# status marking for current state
# sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if it is blinked
    if (ratio > 0.20):
        return 1

    else:
        return 0

def yawn(u1,u2,u3,u4,u5,u6,l1,l2,l3,l4,l5,l6):
    upper_lip_mean = (u1+u2+u3+u4+u5+u6)/6
    lower_lip_mean = (l1+l2+l3+l4+l5+l6)/6
    lip_distance = compute(upper_lip_mean,lower_lip_mean)
    return lip_distance



while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    # detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame =frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        yawn_detect = yawn(landmarks[50],landmarks[51],landmarks[52],
                           landmarks[61], landmarks[62], landmarks[63],
                           landmarks[56], landmarks[57], landmarks[58],
                           landmarks[65], landmarks[66], landmarks[67] )

        # Now judge what to do for the eye blinks
        if (left_blink == 0 or right_blink == 0 or yawn_detect>20):
            drowsy += 1
            active = 0
            if (drowsy > 6):
                status = "Drowsy !!!"
                color = (0, 0, 255)
                if (left_blink == 0 or right_blink == 0):
                    alarm1()
                if(yawn_detect>20):
                    alarm2()



        else:
            drowsy = 0
            active += 1
            if (active > 6):
                status = "Active :)"
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        cv2.imshow("Result of detector", frame)
        cv2.imshow("Frame", face_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
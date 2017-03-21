""" Simple example code for reading an image from a webcam
and manipulating pixels within the resulting image """

import numpy as np
import cv2wrap

cap = cv2wrap.VideoCapture(0)

while (True):
    # capture a frame
    ret, img = cap.read()

    # OUr operations on the frame come here
    gray = cv2wrap.cvtColor(img, cv2wrap.COLOR_BGR2GRAY)

    #gray[1:200:2, 1:200:2] = 0;

    cv2wrap.imshow('img', gray)

    if cv2wrap.waitKey(1) & 0xFF == ord('q'):
        break

# release everything
cap.release()
cv2wrap.destroyAllWindows()

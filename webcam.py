# created by Huang Lu
# 27/08/2016 17:05:45
# Department of EE, Tsinghua Univ.

import cv2
import sys
import numpy as np


def local_present():
    cap = cv2.VideoCapture(0)
    while True:
        # get a frame
        ret, frame = cap.read()
        # show a frame
        if ret:
            cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    local_present()
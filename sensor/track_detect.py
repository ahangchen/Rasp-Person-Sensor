# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import warnings
import json
import datetime
import imutils
import time

import cv2
import uuid
import os

from keras.applications.mobilenet import preprocess_input, relu6, DepthwiseConv2D
from keras.models import load_model


class TempImage:
    def __init__(self, basePath="./", ext=".jpg"):
        # construct the file path
        self.path = "{base_path}/{rand}{ext}".format(base_path=basePath,
                                                     rand=str(uuid.uuid4()), ext=ext)

    def cleanup(self):
        # remove the file
        os.remove(self.path)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="json config file path")
args = vars(ap.parse_args())
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(conf["camera_warmup_time"])
fps = FPS().start()

avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0
found = False

net = load_model('mbp.h5', custom_objects={'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D})

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    if frame is None:
        continue
    timestamp = datetime.datetime.now()
    found = False

    frame = imutils.resize(frame, width=500)
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if avg is None:
        print("[INFO] starting background model ...")
        avg = gray.copy().astype("float")
        continue
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue

        img = imutils.resize(frame, width=224, height=224)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        y = net.predict(img)

        if np.argmax(y) == 1:
            found = True
            # compute the bounding box for the contour, draw it on the frame, and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if found:
        if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
            # increment the motion counter
            motionCounter += 1

            # check to see if the number of frames with consistent motion is high enough
            if motionCounter >= conf["min_motion_frames"]:
                if conf["upload"]:
                    t = TempImage()
                    cv2.imwrite(t.path, frame)

                    # upload the image to server and cleanup the tempory image
                    print("upload {}".format(timestamp))
                    img_upload_url = conf["img_upload_url"]
                    # post image
                    t.cleanup()
                # update the last uploaded time stamp and reset the motion counter
                lastUploaded = timestamp
                motionCounter = 0
    else:
        motionCounter = 0

    if conf["show_video"]:
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

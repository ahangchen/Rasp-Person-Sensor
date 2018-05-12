import os
import cv2
import uuid

import time

from backend.api import ApiConfig
from backend.util import upload_file, post_json, post_form


class TempImage:
    def __init__(self, basePath="./", ext=".jpg"):
        # construct the file path
        self.path = "{base_path}/{rand}{ext}".format(base_path=basePath,
                                                     rand=str(uuid.uuid4()), ext=ext)

    def cleanup(self):
        # remove the file
        os.remove(self.path)



def upload_detection_info(frame, boxes, sensorId):
    t = TempImage()
    cv2.imwrite(t.path, frame)
    svr_conf = ApiConfig()
    img_url = upload_file(svr_conf.urls['upload_img'], t.path)
    detect_json = {
        "captureTime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fromSensorId": sensorId,
        "imgPath": img_url,
        "boxes": []
    }
    for box in boxes:
        detect_json["boxes"].append(box)

    print post_json(svr_conf.urls['upload_detect_info'], detect_json)
    # post image
    t.cleanup()


def upload_wifi_info(mac, rssi, sensorId, time_stamp):
    svr_conf = ApiConfig()
    wifi_json = {
        "captureTime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_stamp)),
        "fromSensorId": sensorId,
        "macAddress": mac,
        "intensity": rssi
    }
    print post_form(svr_conf.urls['upload_wifi_info'], wifi_json)

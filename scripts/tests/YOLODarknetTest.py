#!/envs/drAIver/bin/python
from darknet import *
import cv2
import draiver.util.drawing as dr

# Tiny YOLOv3
NET = b"cfg/yolov3-tiny.cfg"
WEIGHTS = b"bin/yolov3-tiny.weights"

# YOLOv3
# NET = b"cfg/yolov3.cfg"
# WEIGHTS = b"bin/yolov3.weights"

DARKNET_PATH = b"/Users/marco/Documents/GitProjects/UNIVE/darknet/"
DARKNET_PATH_NO_BIN = "/Users/marco/Documents/GitProjects/UNIVE/darknet/"

def main_noopencv(net, meta):
    r = detect(net, meta, DARKNET_PATH + b"data/dog.jpg")
    print(r)

def main_opencv(net, meta):
    imgcv = cv2.imread(DARKNET_PATH_NO_BIN + "data/dog.jpg")
    im = nparray_to_image(imgcv)
    r = detect_im(net, meta, im)
    detections = convert_format(r)

    for res in detections:
        dr.draw_detection(imgcv, res)

    cv2.imshow("w", imgcv)
    cv2.waitKey(0)

    print(detections)

def convert_format(r):
    detections = []
    for obj_class, obj_confidence, (x, y, width, height) in r:
        offset_x = width/2.0
        offset_y = height/2.0
        detection = {'label': obj_class.decode("utf-8") , 'confidence': float(obj_confidence), 'topleft': {'x': float(x - offset_x), 'y': float(y - offset_y)}, 'bottomright': {'x': float(x + offset_x), 'y': float(y + offset_y)}}
        detections.append(detection)
    return detections


if __name__ == "__main__":
    net = load_net(DARKNET_PATH + NET, DARKNET_PATH + WEIGHTS, 0)
    meta = load_meta(DARKNET_PATH + b"cfg/coco.data")  # TODo add absolute paths otherwise it doesn't work

    main_noopencv(net, meta)
    main_opencv(net, meta)




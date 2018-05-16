#!/envs/drAIver/bin/python
from darknet import *
import cv2
import draiver.util.drawing as dr
from os import listdir
from os.path import isfile, join

# Tiny YOLOv3
# NET = b"cfg/yolov3-tiny.cfg"
# WEIGHTS = b"bin/yolov3-tiny.weights"

NET = b"cfg/tiny-yolov3-kitty.cfg"
WEIGHTS = b"bin/tiny-yolov3-kitty.weights"
DATA = b"cfg/kitty.data"

# YOLOv3
# NET = b"cfg/yolov3.cfg"
# WEIGHTS = b"bin/yolov3.weights"

IMAGES = "/Users/marco/Documents/Datasets/drAIver/KITTY/data_object_image_2/testing/image_2/"

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

def opencv_test(net, meta, img):
    im = nparray_to_image(img)
    r = detect_im(net, meta, im)
    detections = convert_format(r)

    for res in detections:
        dr.draw_detection(img, res)

    cv2.imshow("w", img)
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


def main():
    images = [f for f in listdir(IMAGES) if isfile(join(IMAGES, f))]
    images = [f for f in images if ".png" in f]
    images = [f for f in images if '._' not in f]  # Avoid mac issues

    net = load_net(DARKNET_PATH + NET, DARKNET_PATH + WEIGHTS, 0)
    meta = load_meta(DARKNET_PATH + DATA)

    for filename in images:
        img = cv2.imread(IMAGES + filename)
        opencv_test(net, meta, img)



if __name__ == "__main__":
    main()
    # net = load_net(DARKNET_PATH + NET, DARKNET_PATH + WEIGHTS, 0)
    # meta = load_meta(DARKNET_PATH + DATA)  # TODo add absolute paths otherwise it doesn't work
    #
    # main_noopencv(net, meta)
    # main_opencv(net, meta)




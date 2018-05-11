#!/envs/drAIver/bin/python
from darknet import *
import cv2


DARKNET_PATH = b"/Users/marco/Documents/GitProjects/UNIVE/darknet/"
DARKNET_PATH_NO_BIN = "/Users/marco/Documents/GitProjects/UNIVE/darknet/"

def main_noopencv(net, meta):
    r = detect(net, meta, DARKNET_PATH + b"data/dog.jpg")
    print(r)

def main_opencv(net, meta):
    imgcv = cv2.imread(DARKNET_PATH_NO_BIN + "data/dog.jpg")
    im = nparray_to_image(imgcv)
    r = detect_im(net, meta, im)

    print(r)


if __name__ == "__main__":
    net = load_net(DARKNET_PATH + b"cfg/yolov3-tiny.cfg", DARKNET_PATH + b"bin/yolov3-tiny.weights", 0)
    meta = load_meta(DARKNET_PATH + b"cfg/coco.data")  # TODo add absolute paths otherwise it doesn't work

    main_noopencv(net, meta)
    main_opencv(net, meta)




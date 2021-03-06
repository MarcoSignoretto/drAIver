#!/envs/drAIver/bin/python
from darknet import *
import cv2
import draiver.util.drawing as dr
from os import listdir
from os.path import isfile, join
import draiver.env as env
import draiver.camera.properties as cp

# COCO
# NET_KITTY = b"cfg/tiny-yolov3-coco.cfg"
# WEIGHTS_KITTY = b"bin/tiny-yolov3-coco.weights"
# DATA_KITTY = b"cfg/coco.data"

# KITTY
NET_KITTY = b"cfg/tiny-yolov3-kitty.cfg"
WEIGHTS_KITTY = b"bin/tiny-yolov3-kitty.weights"
DATA_KITTY = b"cfg/kitty.data"

# LISA
NET_LISA = b"cfg/tiny-yolov3-lisa.cfg"
WEIGHTS_LISA = b"bin/tiny-yolov3-lisa.weights"
DATA_LISA = b"cfg/lisa.data"

# YOLOv3
# NET = b"cfg/yolov3.cfg"
# WEIGHTS = b"bin/yolov3.weights"

DATASETS_PATH = env.DATASETS_HOME

IMAGES = DATASETS_PATH+"kitty_train_test/images_train/"

DARKNET_PATH = env.DARKNET_HOME
DARKNET_PATH_NO_BIN = env.DARKNET_HOME_NO_BIN

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

def opencv_test(net, meta, img, threshold, wait=0):
    im = nparray_to_image(img)
    r = detect_im(net, meta, im, thresh=threshold)
    detections = convert_format(r)

    for res in detections:
        dr.draw_detection(img, res)

    cv2.imshow("w", img)
    cv2.waitKey(wait)

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
    THRESHOLD = 0.1

    images = [f for f in listdir(IMAGES) if isfile(join(IMAGES, f))]
    images = [f for f in images if ".png" in f]
    images = [f for f in images if '._' not in f]  # Avoid mac issues

    net = load_net(DARKNET_PATH + NET_KITTY, DARKNET_PATH + WEIGHTS_KITTY, 0)
    meta = load_meta(DARKNET_PATH + DATA_KITTY)

    for filename in images:
        img = cv2.imread(IMAGES + filename)
        opencv_test(net, meta, img, threshold=THRESHOLD)

def main_camera():
    THRESHOLD = 0.1

    net_lisa = load_net(DARKNET_PATH + NET_LISA, DARKNET_PATH + WEIGHTS_LISA, 0)
    meta_lisa = load_meta(DARKNET_PATH + DATA_LISA)

    net_kitty = load_net(DARKNET_PATH + NET_KITTY, DARKNET_PATH + WEIGHTS_KITTY, 0)
    meta_kitty = load_meta(DARKNET_PATH + DATA_KITTY)

    vc = cv2.VideoCapture(1)
    print(vc.set(cv2.CAP_PROP_FRAME_WIDTH, cp.FRAME_WIDTH))
    print(vc.set(cv2.CAP_PROP_FRAME_HEIGHT, cp.FRAME_HEIGHT))
    print(vc.set(cv2.CAP_PROP_FPS, cp.FPS))

    while True:
        _, frame = vc.read()

        # put frame into input queue of interested thread
        opencv_test(net_kitty, meta_kitty, frame, threshold=THRESHOLD, wait=1)
        #opencv_test(net_lisa, meta_lisa, frame, threshold=THRESHOLD, wait=1)


if __name__ == "__main__":
    # main()
    main_camera()
    # net = load_net(DARKNET_PATH + NET, DARKNET_PATH + WEIGHTS, 0)
    # meta = load_meta(DARKNET_PATH + DATA)  # TODo add absolute paths otherwise it doesn't work
    #
    # main_noopencv(net, meta)
    # main_opencv(net, meta)




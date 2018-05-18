#!/envs/drAIver/bin/python
from darknet import *

# Deprecated
# DARKFLOW_BASE = "/Users/marco/Documents/GitProjects/UNIVE/darkflow/" # MAC
# DARKFLOW_BASE = "" # Ubuntu

# Meta graph setup for sign
# SIGN_NET_PB_GRAPH = DARKFLOW_BASE+"built_graph/lisa/tiny-yolov2-lisa.pb"
# SIGN_NET_PB_META = DARKFLOW_BASE+"built_graph/lisa/tiny-yolov2-lisa.meta"
#
# # Meta grap setup for car, pedestrians, bicycle
# CAR_NET_PB_GRAPH = DARKFLOW_BASE+"built_graph/kitty/tiny-yolov2-kitty.pb"
# CAR_NET_PB_META = DARKFLOW_BASE+"built_graph/kitty/tiny-yolov2-kitty.meta"
#
# TEST_NET_MODEL = DARKFLOW_BASE+"cfg/tiny-yolo-voc.cfg"
# TEST_NET_WEIGHTS = DARKFLOW_BASE+"bin/tiny-yolo-voc.weights"

# MAC
DARKNET_PATH = b"/Users/marco/Documents/GitProjects/UNIVE/darknet/"

# KITTY
KITTY_NET = b"cfg/tiny-yolov3-kitty.cfg"
KITTY_WEIGHTS = b"bin/tiny-yolov3-kitty.weights"
KITTY_DATA = b"cfg/kitty.data"

# LISA
LISA_NET = b"cfg/tiny-yolov3-kitty.cfg"
LISA_WEIGHTS = b"bin/tiny-yolov3-kitty.weights"
LISA_DATA = b"cfg/kitty.data"


class ObjectDetector:

    def __init__(self, net, weights, data, threshold=0.5):
        self.threshold = threshold
        self.net = load_net(DARKNET_PATH + net, DARKNET_PATH + weights, 0)
        self.meta = load_meta(DARKNET_PATH + data)

    def convert_format(self, r):
        detections = []
        for obj_class, obj_confidence, (x, y, width, height) in r:
            offset_x = width / 2.0
            offset_y = height / 2.0
            detection = {'label': obj_class.decode("utf-8"), 'confidence': float(obj_confidence),
                         'topleft': {'x': float(x - offset_x), 'y': float(y - offset_y)},
                         'bottomright': {'x': float(x + offset_x), 'y': float(y + offset_y)}}
            detections.append(detection)
        return detections

    def detect(self, frame):
        im = nparray_to_image(frame)
        r = detect_im(self.net, self.meta, im, thresh=self.threshold)
        return self.convert_format(r)

    def detect_im(self, im):
        return detect_im(self.net, self.meta, im, thresh=self.threshold)


class SignDetector(ObjectDetector):

    def __init__(self, net=LISA_NET, weights=LISA_WEIGHTS, data=LISA_DATA, threshold=0.5):
        super().__init__(net, weights, data, threshold)


class CarDetector(ObjectDetector):
    def __init__(self, net=KITTY_NET, weights=KITTY_WEIGHTS, data=KITTY_DATA, threshold=0.5):
        super().__init__(net, weights, data, threshold)

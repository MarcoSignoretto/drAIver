#!/envs/drAIver/bin/python
from darkflow.net.build import TFNet

DARKFLOW_BASE = "/Users/marco/Documents/GitProjects/UNIVE/darkflow/" # MAC
# DARKFLOW_BASE = "" # Ubuntu


# Meta graph setup for sign
SIGN_NET_PB_GRAPH = DARKFLOW_BASE+"built_graph/lisa/tiny-yolov2-lisa.pb"
SIGN_NET_PB_META = DARKFLOW_BASE+"built_graph/lisa/tiny-yolov2-lisa.meta"

# Meta grap setup for car, pedestrians, bicycle
CAR_NET_PB_GRAPH = DARKFLOW_BASE+"built_graph/kitty/tiny-yolov2-kitty.pb"
CAR_NET_PB_META = DARKFLOW_BASE+"built_graph/kitty/tiny-yolov2-kitty.meta"

TEST_NET_MODEL = DARKFLOW_BASE+"cfg/tiny-yolo-voc.cfg"
TEST_NET_WEIGHTS = DARKFLOW_BASE+"bin/tiny-yolo-voc.weights"

class ObjectDetector:

    def __init__(self, net_pb_graph, meta_data, threshold=0.2):
        # TODO fixme
        # options = {"pbLoad": net_pb_graph, "metaLoad": meta_data, "threshold": threshold}
        options = {"model": TEST_NET_MODEL, "load": TEST_NET_WEIGHTS, "threshold": 0.2}
        self.detector = TFNet(options)

    def detect(self, frame):
        return self.detector.return_predict(frame)


class SignDetector(ObjectDetector):

    def __init__(self, sign_net_pb_graph=SIGN_NET_PB_GRAPH, threshold=0.2):
        super().__init__(sign_net_pb_graph, SIGN_NET_PB_META, threshold)


class CarDetector(ObjectDetector):
    def __init__(self, car_net_pb_graph=CAR_NET_PB_GRAPH, threshold=0.2):
        super().__init__(car_net_pb_graph, CAR_NET_PB_META, threshold)

#!/envs/drAIver/bin/python
from darkflow.net.build import TFNet
import cv2
import draiver.env as env

DARKFLOW_BASE = env.DATASETS_HOME

# tiny-yolo-voc has same architecture of YOLOv2 with some differences
TEST_NET_MODEL = DARKFLOW_BASE+"cfg/tiny-yolo-voc.cfg"
TEST_NET_WEIGHTS = DARKFLOW_BASE+"bin/tiny-yolo-voc.weights"

# Meta graph setup for sign
SIGN_NET_PB_GRAPH = DARKFLOW_BASE+"built_graph/lisa/tiny-yolov2-lisa.pb"
SIGN_NET_PB_META = DARKFLOW_BASE+"built_graph/lisa/tiny-yolov2-lisa.meta"

# Meta grap setup for car, pedestrians, bicycle
CAR_NET_PB_GRAPH = DARKFLOW_BASE+"built_graph/kitty/tiny-yolov2-kitty.pb"
CAR_NET_PB_META = DARKFLOW_BASE+"built_graph/kitty/tiny-yolov2-kitty.meta"

# NET_MODEL = DARKFLOW_BASE+"cfg/tiny-yolo-new.cfg" # TODO remove is test on training

# Meta graph setup
# NET_PB_GRAPH = DARKFLOW_BASE+"built_graph/tiny-yolo-new.pb"
# NET_PB_META = DARKFLOW_BASE+"built_graph/tiny-yolo-new.meta"

# BASE_PATH = "/mnt/B01EEC811EEC41C8/" # Ubuntu Config
BASE_PATH = env.DATASETS_HOME


def main():
    images = [
        #"object_detector/test_images/car.png",
        #"object_detector/test_images/car_2.png",
        "object_detector/test_images/cars.png",
    ]

    options = {"model": TEST_NET_MODEL, "load": TEST_NET_WEIGHTS, "threshold": 0.2}

    #protobuf load
    #options = {"pbLoad": CAR_NET_PB_GRAPH, "metaLoad": CAR_NET_PB_META, "threshold": 0.0}

    tfnet = TFNet(options)

    for path in images:

        # imgcv = cv2.imread(BASE_PATH+"KITTY/data_object_image_2/training/image_2/000010.png")
        imgcv = cv2.imread(BASE_PATH+path)
        result = tfnet.return_predict(imgcv)
        for res in result:
            pt1 = (res['topleft']['x'], res['topleft']['y'])
            pt2 = (res['bottomright']['x'], res['bottomright']['y'])
            cv2.rectangle(imgcv, pt1, pt2, (0, 255, 0), thickness=3, lineType=cv2.LINE_8)

        print(result)
        cv2.imshow("Img", imgcv)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

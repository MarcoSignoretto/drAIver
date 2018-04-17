#!/envs/drAIver/bin/python
from darkflow.net.build import TFNet
import cv2

DARKFLOW_BASE = "/Users/marco/Documents/GitProjects/UNIVE/darkflow/" # MAC
# DARKFLOW_BASE = "" # Ubuntu

# tiny-yolo-voc has same architecture of YOLOv2 with some differences
TEST_NET_MODEL = DARKFLOW_BASE+"cfg/tiny-yolo-voc.cfg"
TEST_NET_WEIGHTS = DARKFLOW_BASE+"bin/tiny-yolo-voc.weights"

# NET_MODEL = DARKFLOW_BASE+"cfg/tiny-yolo-new.cfg" # TODO remove is test on training

# Meta graph setup
# NET_PB_GRAPH = DARKFLOW_BASE+"built_graph/tiny-yolo-new.pb"
# NET_PB_META = DARKFLOW_BASE+"built_graph/tiny-yolo-new.meta"

# BASE_PATH = "/mnt/B01EEC811EEC41C8/" # Ubuntu Config
BASE_PATH = "/Users/marco/Documents/"


def main():
    images = [
        "Datasets/drAIver/object_detector/test_images/car.png",
        "Datasets/drAIver/object_detector/test_images/car_2.png",
        "Datasets/drAIver/object_detector/test_images/cars.png",
    ]

    options = {"model": TEST_NET_MODEL, "load": TEST_NET_WEIGHTS, "threshold": 0.2}

    #protobuf load
    # options = {"pbLoad": NET_PB_GRAPH, "metaLoad": NET_PB_META, "threshold": 0.16}

    tfnet = TFNet(options)

    for path in images:

        # imgcv = cv2.imread(BASE_PATH+"Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000010.png")
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

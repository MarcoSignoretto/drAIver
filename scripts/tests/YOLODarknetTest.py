#!/envs/drAIver/bin/python
from darknet import *

DARKNET_PATH = b"/Users/marco/Documents/GitProjects/UNIVE/darknet/"

if __name__ == "__main__":
    # net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    # im = load_image("data/wolf.jpg", 0, 0)
    # meta = load_meta("cfg/imagenet1k.data")
    # r = classify(net, meta, im)
    # print r[:10]
    net = load_net(DARKNET_PATH+b"cfg/yolov3-tiny.cfg", DARKNET_PATH+b"bin/yolov3-tiny.weights", 0)
    meta = load_meta(DARKNET_PATH+b"cfg/coco.data") # TODo add absolute paths otherwise it doesn't work
    r = detect(net, meta, DARKNET_PATH+b"data/dog.jpg")
    print(r)


#!/envs/drAIver/bin/python
import sys, getopt
from os import listdir
from os.path import isfile, join
from draiver.detectors.objectdetector import SignDetector
from draiver.detectors.objectdetector import CarDetector
from lxml import etree
import cv2
import numpy as np

BASE_PATH = "/Users/marco/Documents/" # Mac Config
# BASE_PATH = "/mnt/B01EEC811EEC41C8/" # Ubuntu Config

DATASETS_PATH = BASE_PATH + "GitProjects/UNIVE/darkflow/training/"

# TODO fix kitty test images ( not available )

# =========== MAC configs =======
KITTY_PATH = DATASETS_PATH + "kitty/"
LISA_PATH = DATASETS_PATH + "lisa_train_test/"

KITTY_TEST_IMAGES = KITTY_PATH + "images/" # TODO complete
LISA_TEST_IMAGES = LISA_PATH + "images_test/" # TODO complete

KITTY_GROUND_TRUTH = KITTY_PATH + "annotations/" # TODO complete
LISA_GROUND_TRUTH = LISA_PATH + "annotations_test/" # TODO complete

def compute_score_components(detector, test_images_path, gt_path):
    # TODO complete
    return None, None, None, None


def find_box_detection(detection):
    return np.array([
        float(detection['topleft']['x']),
        float(detection['topleft']['y']),
        float(detection['bottomright']['x']),
        float(detection['bottomright']['y'])
    ])


def find_class_detection(detection):
    return str(detection['label'])


def find_box_gt(xml_obj):
    xmin = float(xml_obj.find('bndbox').find('xmin').text)
    xmax = float(xml_obj.find('bndbox').find('xmax').text)
    ymin = float(xml_obj.find('bndbox').find('ymin').text)
    ymax = float(xml_obj.find('bndbox').find('ymax').text)
    return np.array([
        xmin,
        ymin,
        xmax,
        ymax
    ])


def find_class_gt(xml_obj):
    return str(xml_obj.find('name'))


def compute_IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def compute_frame_IoU(mapping):
    IoU_List = []
    for (detection, annotation) in mapping:
        if detection is not None and annotation is not None:
            boxA = find_box_detection(detection)
            boxB = find_box_gt(annotation)
            IoU_List.append(compute_IoU(boxA, boxB))

    return np.mean(IoU_List)


def compute_frame_statistics(mapping):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for (detection, annotation) in mapping:
        if detection is None and annotation is not None: # FN
            FN = FN + 1
        elif detection is not None and annotation is not None: # TP
            det_class = find_class_detection(detection)
            ann_class = find_class_gt(annotation)
            if det_class == ann_class:
                TP = TP + 1
            else:
                FN =
        elif detection is not None and annotation is None: # FP
            FP = FP + 1



#
# def compute_frame_iou(detections, gt_path):
#     root = etree.parse(gt_path)
#
#     objects = root.findall("object")
#
#     # TODO remove
#     map = find_mapping(detections, objects)
#
#
#
#
#     IoU_list = []
#     for detection in detections:
#         max_iou = 0.0
#         boxA = find_box_detection(detection)
#         detectionClass = find_class_detection(detection)
#         for obj in objects:
#             boxB = find_box_gt(obj)
#             objClass = find_class_gt(obj)
#             if detectionClass == objClass:
#                 iou = compute_IoU(boxA, boxB)
#                 if iou > max_iou:
#                     max_iou = iou
#         if max_iou > 0.0:
#             IoU_list.append(max_iou)
#
#     return np.mean(IoU_list)


def find_mapping(detections, objects):
    detection_object_mapping = []
    for detection in detections:
        max_iou = 0.0
        boxA = find_box_detection(detection)

        bestObject = None
        for obj in objects:

            boxB = find_box_gt(obj)
            iou = compute_IoU(boxA, boxB)
            if iou > max_iou:
                max_iou = iou
                bestObject = obj

        detection_object_mapping.append((detection, bestObject))

    for obj in objects:
        present = False
        for det, annotation in detection_object_mapping:
            if obj == annotation:
                present = True
        if not present:
            detection_object_mapping.append((None, obj))

    return detection_object_mapping



def mean_iou(detector, test_images_path, gt_path):
    images = [f for f in listdir(test_images_path) if isfile(join(test_images_path, f))]
    images = [f for f in images if ".png" in f]

    avgIoU = 0.0
    for filename in images:
        filenameXml = filename.replace('.png', '.xml')

        img = cv2.imread(str(test_images_path) + str(filename))
        detections = detector.detect(img)
        avgIoU = avgIoU + compute_frame_iou(detections, gt_path + filenameXml)

    return avgIoU / float(len(images))

def evaluate_frame_performance(detections, annotations):
    mapping = find_mapping(detections, annotations)

    compute_frame_IoU(mapping)



    return ...


def evaluate_performances(detector, test_images_path, gt_path):
    images = [f for f in listdir(test_images_path) if isfile(join(test_images_path, f))]
    images = [f for f in images if ".png" in f]

    for filename in images:
        filenameXml = filename.replace('.png', '.xml')

        img = cv2.imread(str(test_images_path) + str(filename))
        detections = detector.detect(img)

        root = etree.parse(gt_path + filenameXml)
        annotations = root.findall("object")

        performance = evaluate_frame_performance(detections, annotations)

        # TODO complete splitting performances

        # ============ PRINT =============

        for detection in detections:
            pt1 = (detection['topleft']['x'], detection['topleft']['y'])
            pt2 = (detection['bottomright']['x'], detection['bottomright']['y'])
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=3, lineType=cv2.LINE_8)

        for annotation in annotations:
            box = find_box_gt(annotation)
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(img, pt1, pt2, (255, 0, 0), thickness=2, lineType=cv2.LINE_8)

        print(mapping)
        for det, gt in mapping:
            if det is not None:
                pt1 = (det['topleft']['x'], det['topleft']['y'])
                pt2 = (det['bottomright']['x'], det['bottomright']['y'])
                cv2.rectangle(img, pt1, pt2, (0, 0, 255), thickness=1, lineType=cv2.LINE_8)

            if gt is not None:
                box = find_box_gt(gt)
                pt1 = (int(box[0]), int(box[1]))
                pt2 = (int(box[2]), int(box[3]))
                cv2.rectangle(img, pt1, pt2, (0, 13, 55), thickness=1, lineType=cv2.LINE_8)

        cv2.imshow("Frame", img)
        cv2.waitKey(0)


def main(option, test_images_path, gt_path):
    threshold = 0.1
    results = []

    while threshold <= 1.0:

        if option == "kitty":
            detector = CarDetector(threshold=threshold)
        elif option == "lisa":
            detector = SignDetector(threshold=threshold)
        else:
            print("ERROR!!!!!!! invalid option!!!!")

        results.append(evaluate_performances(detector, test_images_path, gt_path))

        threshold = threshold + 0.1

    # ===================== Report summary =========================


if __name__ == '__main__':

    test_images_path = None
    gt_path = None
    option = None
    num_frames = 100
    display = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:', ['dataset='])
    except getopt.GetoptError:
        print('objectdetector_evaluation.py -d <kitty/lisa>')
        sys.exit(2)

    for o, a in opts:
        if o in ("-d", "--dataset"):
            if a == "kitty":
                test_images_path = KITTY_TEST_IMAGES
                gt_path = KITTY_GROUND_TRUTH
                option = a
            elif a == "lisa":
                test_images_path = LISA_TEST_IMAGES
                gt_path = LISA_GROUND_TRUTH
                option = a
        else:
            assert False, "unhandled option"

    main(option, test_images_path, gt_path)


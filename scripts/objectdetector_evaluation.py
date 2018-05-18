#!/envs/drAIver/bin/python
import sys, getopt
from os import listdir
from os.path import isfile, join
from draiver.detectors.objectdetector import SignDetector
from draiver.detectors.objectdetector import CarDetector
from lxml import etree
import cv2
import numpy as np
import draiver.util.drawing as dr
import draiver.util.detectorutil as du
import time
from darknet import *

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

IOU_THRESHOLD = 0.5

def compute_score_components(detector, test_images_path, gt_path):
    # TODO complete
    return None, None, None, None

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
            boxA = du.find_box_detection(detection)
            boxB = du.find_box_gt(annotation)
            IoU_List.append(compute_IoU(boxA, boxB))

    return np.mean(IoU_List)


def compute_frame_statistics(mapping):
    TP = 0
    FP = 0
    FN = 0
    for (detection, annotation) in mapping:
        if detection is None and annotation is not None: # FN
            FN = FN + 1
        elif detection is not None and annotation is not None: # TP
            det_class = du.find_class_detection(detection)
            ann_class = du.find_class_gt(annotation)
            if det_class == ann_class: # Only if class is correct we have a true positive otherwise a false negative
                TP = TP + 1
            else:
                FN = FN + 1
        elif detection is not None and annotation is None: # FP
            FP = FP + 1

    return TP, FP, FN




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
        # Kitty skip dontcare class
        dclass = du.find_class_detection(detection)
        if dclass != 'DontCare':
            max_iou = 0.0
            boxA = du.find_box_detection(detection)

            bestObject = None
            for obj in objects:

                boxB = du.find_box_gt(obj)
                iou = compute_IoU(boxA, boxB)
                if iou > max_iou and iou > IOU_THRESHOLD: # TODO evaluate if ok set iou threshold
                    max_iou = iou
                    bestObject = obj

            detection_object_mapping.append((detection, bestObject))

    for obj in objects:
        # Kitty skip dontcare class
        aclass = du.find_class_gt(obj)
        if aclass != 'DontCare':
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
    TP, FP, FN = compute_frame_statistics(mapping)

    return TP, FP, FN


def evaluate_performances(detector, test_images_path, gt_path):
    images = [f for f in listdir(test_images_path) if isfile(join(test_images_path, f))]
    images = [f for f in images if ".png" in f]
    images = [f for f in images if '._' not in f]  # Avoid mac issues

    final_TP = 0
    final_FP = 0
    final_FN = 0
    cumulative_eval_time = 0.0
    count = 0

    for filename in images:
        filenameXml = filename.replace('.png', '.xml')

        img = cv2.imread(str(test_images_path) + str(filename))

        im = nparray_to_image(img)
        t0 = time.process_time()
        r = detector.detect_im(im)
        eval_time = time.process_time() - t0,
        detections = detector.convert_format(r)

        root = etree.parse(gt_path + filenameXml)
        annotations = root.findall("object")

        TP, FP, FN = evaluate_frame_performance(detections, annotations)
        final_TP = final_TP + TP
        final_FP = final_FP + FP
        final_FN = final_FN + FN
        cumulative_eval_time = cumulative_eval_time + eval_time[0]

        # TODO complete splitting performances

        # ============ PRINT =============

        for annotation in annotations:
            aclass = du.find_class_gt(annotation)
            if aclass != 'DontCare':
                dr.draw_gt(img, annotation)

        for detection in detections:
            dclass = du.find_class_detection(detection)
            if dclass != 'DontCare':
                dr.draw_detection(img, detection)

        # TODO remove
        mapping = find_mapping(detections, annotations)
        print(mapping)
        for det, gt in mapping:
            if det is not None:
                pt1 = (int(det['topleft']['x']), int(det['topleft']['y']))
                pt2 = (int(det['bottomright']['x']), int(det['bottomright']['y']))
                cv2.rectangle(img, pt1, pt2, (0, 0, 255), thickness=1, lineType=cv2.LINE_8)

            if gt is not None:
                box = du.find_box_gt(gt)
                pt1 = (int(box[0]), int(box[1]))
                pt2 = (int(box[2]), int(box[3]))
                cv2.rectangle(img, pt1, pt2, (0, 13, 55), thickness=1, lineType=cv2.LINE_8)

        cv2.imshow("Frame", img)
        cv2.waitKey(1)

        count = count + 1
        print("%s %s" % (count, filename))

    # =============== FINAL EVALUATION ================

    precision = float(final_TP) / float(final_TP + final_FP)
    recall = float(final_TP) / float(final_TP + final_FN)
    f1_score = 2 * (precision * recall)/(precision + recall)
    mean_eval_time = cumulative_eval_time / len(images)
    print(precision)
    print(recall)
    print(f1_score)
    print(mean_eval_time)

    return precision, recall, f1_score, mean_eval_time


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
            exit()

        results.append(evaluate_performances(detector, test_images_path, gt_path))

        threshold = threshold + 0.1

    # ===================== Report summary =========================

    # TODO serialize report


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


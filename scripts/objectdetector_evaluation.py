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
import _pickle as pk

# BASE_PATH = "/Users/marco/Documents/GitProjects/UNIVE/darkflow/training/" # Mac Config
BASE_PATH = "/mnt/B01EEC811EEC41C8/Datasets/drAIver/" # Ubuntu Config

DATASETS_PATH = BASE_PATH

# TODO fix kitty test images ( not available )

KITTY_PATH = DATASETS_PATH + "kitty_train_test/"
LISA_PATH = DATASETS_PATH + "lisa_train_test/"

KITTY_TEST_IMAGES = KITTY_PATH + "images_test/" # TODO complete
LISA_TEST_IMAGES = LISA_PATH + "images_test/" # TODO complete

KITTY_GROUND_TRUTH = KITTY_PATH + "annotations_test/" # TODO complete
LISA_GROUND_TRUTH = LISA_PATH + "annotations_test/" # TODO complete


IOU_THRESHOLD = 0.5
DETECTION_THRESHOLD_BASE = 0.01
DETECTION_THRESHOLD_PACE = 0.01

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
            if det_class == ann_class: # Only if class is correct we have a true positive otherwise a false negative ( class always correct sinc mapping is performend only with single class)
                TP = TP + 1
            else:
                FN = FN + 1
        elif detection is not None and annotation is None: # FP
            FP = FP + 1

    return TP, FP, FN


def find_mapping(detections, objects, object_class):
    detection_object_mapping = []
    for detection in detections:
        # Kitty skip dontcare class
        dclass = du.find_class_detection(detection)
        if dclass == object_class:
            max_iou = 0.0
            boxA = du.find_box_detection(detection)

            bestObject = None
            for obj in objects:
                aclass = du.find_class_gt(obj)
                if aclass == object_class:
                    boxB = du.find_box_gt(obj)
                    iou = compute_IoU(boxA, boxB)
                    if iou > max_iou and iou > IOU_THRESHOLD: # TODO evaluate if ok set iou threshold
                        max_iou = iou
                        bestObject = obj

            detection_object_mapping.append((detection, bestObject))

    for obj in objects:
        # Kitty skip dontcare class
        aclass = du.find_class_gt(obj)
        if aclass == object_class:
            present = False
            for det, annotation in detection_object_mapping:
                if obj == annotation:
                    present = True
            if not present:
                detection_object_mapping.append((None, obj))

    return detection_object_mapping

# TODO remove when mapping divided by class works
# def find_mapping(detections, objects):
#     detection_object_mapping = []
#     for detection in detections:
#         # Kitty skip dontcare class
#         dclass = du.find_class_detection(detection)
#         if dclass != 'DontCare':
#             max_iou = 0.0
#             boxA = du.find_box_detection(detection)
#
#             bestObject = None
#             for obj in objects:
#                 aclass = du.find_class_gt(obj)
#                 if aclass != 'DontCare':
#                     boxB = du.find_box_gt(obj)
#                     iou = compute_IoU(boxA, boxB)
#                     if iou > max_iou and iou > IOU_THRESHOLD: # TODO evaluate if ok set iou threshold
#                         max_iou = iou
#                         bestObject = obj
#
#             detection_object_mapping.append((detection, bestObject))
#
#     for obj in objects:
#         # Kitty skip dontcare class
#         aclass = du.find_class_gt(obj)
#         if aclass != 'DontCare':
#             present = False
#             for det, annotation in detection_object_mapping:
#                 if obj == annotation:
#                     present = True
#             if not present:
#                 detection_object_mapping.append((None, obj))
#
#     return detection_object_mapping


def evaluate_frame_performance(detections, annotations, object_class):
    mapping = find_mapping(detections, annotations, object_class)
    TP, FP, FN = compute_frame_statistics(mapping)
    return TP, FP, FN


def evaluate_performances(detector, test_images_path, gt_path, object_class):
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
        t0 = time.time()
        r = detector.detect_im(im)
        eval_time = time.time() - t0,
        detections = detector.convert_format(r)

        root = etree.parse(gt_path + filenameXml)
        annotations = root.findall("object")

        TP, FP, FN = evaluate_frame_performance(detections, annotations, object_class)
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
        mapping = find_mapping(detections, annotations, object_class)
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

    precision = float(final_TP) / float(final_TP + final_FP) if float(final_TP + final_FP) != 0.0 else np.nan
    recall = float(final_TP) / float(final_TP + final_FN) if float(final_TP + final_FN) != 0.0 else np.nan
    f1_score = 2 * (precision * recall)/(precision + recall) if (precision + recall) != 0.0 and precision is not np.nan and recall is not np.nan else np.nan
    mean_eval_time = cumulative_eval_time / len(images)
    print(precision)
    print(recall)
    print(f1_score)
    print(mean_eval_time)

    return precision, recall, f1_score, mean_eval_time


def main(option, dataset_path, test_images_path, gt_path):
    report = {}

    classes = []
    classes_file_path = dataset_path + "labels.txt"
    classes_files = open(classes_file_path).read().strip().split()
    for c in classes_files:
        classes.append(c)

    if option == "kitty":
        detector = CarDetector()
    elif option == "lisa":
        detector = SignDetector()
    else:
        print("ERROR!!!!!!! invalid option!!!!")
        exit()

    for c in classes:

        threshold = DETECTION_THRESHOLD_BASE

        print(c)
        results = []

        while threshold <= 1.0:

            detector.set_threshold(threshold=threshold)

            results.append(evaluate_performances(detector, test_images_path, gt_path, c))

            threshold = threshold + DETECTION_THRESHOLD_PACE

        report[c] = results
        print("class %s completed." % c)

    # ===================== Report summary =========================

    with open('report_%s.pickle' % option, 'wb') as handle:
        pk.dump(report, handle)


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
                dataset_path = KITTY_PATH
                test_images_path = KITTY_TEST_IMAGES
                gt_path = KITTY_GROUND_TRUTH
                option = a
            elif a == "lisa":
                dataset_path = LISA_PATH
                test_images_path = LISA_TEST_IMAGES
                gt_path = LISA_GROUND_TRUTH
                option = a
        else:
            assert False, "unhandled option"

    main(option, dataset_path, test_images_path, gt_path)


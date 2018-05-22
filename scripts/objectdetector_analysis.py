#!/envs/drAIver/bin/python
import _pickle as pk
import sys, getopt
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
from lxml import etree
import draiver.util.detectorutil as du
import draiver.env as env

DATASETS_PATH = env.DATASETS_HOME

# TODO fix kitty test images ( not available )

KITTY_PATH = DATASETS_PATH + "kitty_train_test/"
LISA_PATH = DATASETS_PATH + "lisa_train_test/"

KITTY_TEST_IMAGES = KITTY_PATH + "images_test/" # TODO complete
LISA_TEST_IMAGES = LISA_PATH + "images_test/" # TODO complete

KITTY_GROUND_TRUTH = KITTY_PATH + "annotations_train/" # TODO complete
LISA_GROUND_TRUTH = LISA_PATH + "annotations_train/" # TODO complete


def compute_fps(report):
    mean_eval_time_list = []
    for key, measures in report.items():
        for precision, m_recall, f1_score, mean_eval_time in measures:
            if mean_eval_time is not np.nan:
                mean_eval_time_list.append(mean_eval_time)

    return 1.0 / np.mean(mean_eval_time_list)


def add_bbox_areas(bbox_areas, annotations):
    for annotation in annotations:
        label = du.find_class_gt(annotation)
        box = du.find_box_gt(annotation)
        area = du.box_area(box)
        if label in bbox_areas.keys():
            bbox_areas[label].append(area)
        else:
            bbox_areas[label] = [area]


def compute_bbox_avg_area():
    if option == 'kitty':
        gt_path = KITTY_GROUND_TRUTH
    elif option == 'lisa':
        gt_path = LISA_GROUND_TRUTH
    else:
        exit()
    images = [f for f in listdir(gt_path) if isfile(join(gt_path, f))]
    images = [f for f in images if ".xml" in f]
    images = [f for f in images if '._' not in f]  # Avoid mac issues

    bbox_areas = {}

    for filename in images:
        root = etree.parse(gt_path + filename)
        annotations = root.findall("object")
        add_bbox_areas(bbox_areas, annotations)

    for key, value in bbox_areas.items():
        bbox_areas[key] = np.mean(value)

    return bbox_areas


def add_annotation_frequencies(frequencies, annotations):
    for annotation in annotations:
        label = du.find_class_gt(annotation)
        if label in frequencies.keys():
            frequencies[label] = frequencies[label] + 1
        else:
            frequencies[label] = 1

def compute_dataset_gt_objects():
    if option == 'kitty':
        gt_path = KITTY_GROUND_TRUTH
    elif option == 'lisa':
        gt_path = LISA_GROUND_TRUTH
    else:
        exit()
    images = [f for f in listdir(gt_path) if isfile(join(gt_path, f))]
    images = [f for f in images if ".xml" in f]
    images = [f for f in images if '._' not in f]  # Avoid mac issues

    frequencies = {}

    for filename in images:
        root = etree.parse(gt_path + filename)
        annotations = root.findall("object")
        add_annotation_frequencies(frequencies, annotations)
    return frequencies


def calculated_pinterpolated(measures, recall):
    max_prec = 0.0
    for precision, m_recall, f1_score, mean_eval_time in measures:
        if m_recall >= recall:
            if precision is not np.nan and precision > max_prec:
                max_prec = precision
    return max_prec


def compute_AP(report):
    res = {}
    for key, measures in report.items():
        print(key)

        AP_list = []

        recall = 0.0
        while recall <= 1.0:
            p_interpolated = calculated_pinterpolated(measures, recall)
            AP_list.append(p_interpolated)

            recall = recall + 0.1

        AP = np.mean(AP_list)
        res[key] = AP
    return res


def compute_mAP(report, consider_classes):
    class_ap = compute_AP(report)
    mAP_list = []
    for key, value in class_ap.items():
        if key in consider_classes:
            mAP_list.append(value)

    return np.mean(mAP_list)


def precision_recall_curve_plot(report):
    for key, measures in report.items():
        precision, recall, f1_score, mean_eval_time = map(list, zip(*measures))

        plt.title(key)
        plt.scatter(recall, precision)
        plt.show()


def main(option):
    consider_classes = ['car','person', 'Cyclist'] if option == 'kitty' else ['stop','yield', 'pedestrianCrossing'] # TODO check
    with open('output/report_%s_001.pickle' % option, 'rb') as handle:
        report = pk.load(handle)

        precision_recall_curve_plot(report)
        AP = compute_AP(report)
        print("=========================================== SUMMARY ==============================")
        mAP = compute_mAP(report, consider_classes)
        print("Avg fps: %s" % str(compute_fps(report)))
        print("mAP for %s is %s" % (option, mAP))
        print("AP for %s is: " % option)
        print(AP)
        print("DATASET GT OBJECTS")
        print(compute_dataset_gt_objects())
        print("DATASET AVG BBOX AREA")
        print(compute_bbox_avg_area())




if __name__ == '__main__':

    option = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:', ['dataset='])
    except getopt.GetoptError:
        print('objectdetector_evaluation.py -d <kitty/lisa>')
        sys.exit(2)

    for o, a in opts:
        if o in ("-d", "--dataset"):
            if a == "kitty":
                option = a
            elif a == "lisa":
                option = a
        else:
            assert False, "unhandled option"

    main(option)



#!/envs/drAIver/bin/python
import sys, getopt
from draiver.detectors.objectdetector import SignDetector
from draiver.detectors.objectdetector import CarDetector

BASE_PATH = "/Users/marco/Documents/" # Mac Config
# BASE_PATH = "/mnt/B01EEC811EEC41C8/" # Ubuntu Config

KITTY_PATH = "" # TODO complete
LISA_PATH = "" # TODO complete

KITTY_TEST_IMAGES = "" # TODO complete
LISA_TEST_IMAGES = "" # TODO complete

KITTY_GROUND_TRUTH = "" # TODO complete
LISA_GROUND_TRUTH = "" # TODO complete

def compute_score_components(detector, test_images_path, gt_path):
    # TODO complete
    return None, None, None, None


def evaluate_performance(detector, test_images_path, gt_path):

    TP, TN, FP, FN = compute_score_components(detector, test_images_path, gt_path)
    prec = # TODO complete
    recall = # TODO complete
    accuracy = # TODO complete
    f1_score = # TODO complete

    # TODO add IoU and mAP

    return {"prec":prec, "recall":recall, "accuracy":accuracy, "f1_score":f1_score}


def main(option, test_images_path, gt_path):
    threshold = 0.1
    results = []

    while threshold <= 1.0:

        if option == "kitty":
            detector = CarDetector(threshold=threshold)
        elif option == "lisa":
            detector = SignDetector(threshold=threshold)

        results.append(evaluate_performance(detector, test_images_path, gt_path))

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


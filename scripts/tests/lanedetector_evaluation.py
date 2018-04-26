#!/envs/drAIver/bin/python
import draiver.camera.properties as cp
import numpy as np
import cv2
import draiver.detectors.line_detector_v3 as ld
from draiver.camera.birdseye import BirdsEye
from os import listdir
from os.path import isfile, join

# BASE_PATH = "/mnt/B01EEC811EEC41C8/" # Ubuntu Config
BASE_PATH = "/Users/marco/Documents/" # Mac Config

EVALUATION_PATH = BASE_PATH+"Datasets/drAIver/KITTY/lane_evaluation/"
IMAGES = EVALUATION_PATH+"images/"
GROUND_TRUTH = EVALUATION_PATH+"gt/"

def convert_for_ground_true():

    images = [f for f in listdir(IMAGES) if isfile(join(IMAGES, f))]

    for path in images:

        img = cv2.imread(str(IMAGES) + str(path))

        gray = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, gray, 1)

        th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, -15)

        cv2.imshow("Original", img)
        cv2.imshow("Threshold", th2)
        cv2.moveWindow("Threshold", 800, 100)

        cv2.imwrite(str(GROUND_TRUTH) + str(path), th2, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

        cv2.waitKey(0)

    cv2.destroyAllWindows()

def calculate_lane_feature_accuracy(line_features, bird_line):
    x, y = line_features # x and y are refered respect to axis of second order polynomial
    tp = 0
    total = len(x)
    for i in range(0, total):
        if bird_line[int(x[i]), int(y[i])][0] > 0:
            tp = tp + 1

    return float(tp)/float(total)


def calculate_lane_position_deviation(line, bird_line):
    deviation = 0

    for i in range(0, bird_line.shape[0]):
        non_zero, _ = np.nonzero(bird_line[i])
        if len(non_zero) > 0:
            min_val = min(non_zero)
            max_val = max(non_zero)

            predicted = line[0]*(i**2) + line[1]*i + line[2]

            if predicted > max_val:
                deviation = deviation + (predicted - max_val)
            elif predicted < min_val:
                deviation = deviation + (min_val - predicted)

    return float(deviation)/float(bird_line.shape[0])


def evaluate():

    acc_left_list = []
    acc_right_list = []
    detection_left_list = []
    detection_right_list = []
    deviation_left_list = []
    deviation_right_list = []

    width = cp.FRAME_WIDTH
    height = cp.FRAME_HEIGHT

    points = np.float32([
        [
            261,
            326
        ], [
            356,
            326
        ], [
            173,
            478
        ], [
            411,
            478
        ]
    ])
    destination_points = np.float32([
        [
            width / cp.CHESSBOARD_ROW_CORNERS,
            height / cp.CHESSBOARD_COL_CORNERS
        ], [
            width - (width / cp.CHESSBOARD_ROW_CORNERS),
            height / cp.CHESSBOARD_COL_CORNERS
        ], [
            width / cp.CHESSBOARD_ROW_CORNERS,
            height + 200
        ], [
            width - (width / cp.CHESSBOARD_ROW_CORNERS),
            height + 200
        ]
    ])

    M = cv2.getPerspectiveTransform(points, destination_points)
    birdview = BirdsEye(M=M, width=width, height=height)

    # images = [f for f in listdir(IMAGES) if isfile(join(IMAGES, f))]
    images = [
        "2011_09_26_0027_0000000022.png",
        # "2011_09_26_0029_0000000361.png",
        "2011_09_26_0028_0000000037.png",
        "2011_09_26_0027_0000000058.png",
        "2011_09_26_0029_0000000035.png",
    ]

    for path in images:
        img = cv2.imread(str(IMAGES) + str(path))
        img = cv2.resize(img, (width, height))

        left_img = cv2.imread(str(GROUND_TRUTH) +"left_"+str(path))
        left_img = cv2.resize(left_img, (width, height))

        right_img = cv2.imread(str(GROUND_TRUTH) +"right_"+str(path))
        right_img = cv2.resize(right_img, (width, height))

        img = cv2.medianBlur(img, 3)

        bird = birdview.apply(img)

        bird_left_col = birdview.apply(left_img)
        bird_left = np.zeros((bird_left_col.shape[0], bird_left_col.shape[1], 1), dtype=np.uint8)
        cv2.cvtColor(bird_left_col, cv2.COLOR_RGB2GRAY, bird_left, 1)

        bird_right_col = birdview.apply(right_img)
        bird_right = np.zeros((bird_right_col.shape[0], bird_right_col.shape[1], 1), dtype=np.uint8)
        cv2.cvtColor(bird_right_col, cv2.COLOR_RGB2GRAY, bird_right, 1)

        # Line Detection
        left, right, left_features, right_features = ld.detect_verbose(bird, negate=False, robot=False, thin=False)
        print(left, right)
        print(left_features, right_features)

        # ========== Plot ============
        l_x, l_y = left_features
        for i in range(0, len(l_x)):
            cv2.circle(bird_left_col, (int(l_y[i]), int(l_x[i])), 1, (0, 255, 0), thickness=1)

        r_x, r_y = right_features
        for i in range(0, len(r_x) - 1):
            cv2.circle(bird_right_col, (int(r_y[i]), int(r_x[i])), 1, (0, 255, 0), thickness=1)

        # ========== Accuracy ===========
        acc_left = calculate_lane_feature_accuracy(left_features, bird_left)
        print(acc_left)

        acc_left_list.append(acc_left)

        acc_right = calculate_lane_feature_accuracy(right_features, bird_right)
        print(acc_right)

        acc_right_list.append(acc_right)

        # ========== Detections ===========
        detection_left_list.append(left is not None)
        detection_right_list.append(right is not None)

        # ========== Lane position deviation ========
        if left is not None:

            dev_left = calculate_lane_position_deviation(left, bird_left)
            print(dev_left)

            deviation_left_list.append(dev_left)

            # ========== Plot ===========
            for i in range(0, bird_left_col.shape[0]):
                y_fit = left[0] * (i ** 2) + left[1] * i + left[2]
                cv2.circle(bird_left_col, (int(y_fit), i), 1, (0, 0, 255), thickness=1)

        if right is not None:

            dev_right = calculate_lane_position_deviation(right, bird_right)
            print(dev_right)

            deviation_right_list.append(dev_right)

            # ========== Plot ===========
            for i in range(0, bird_right_col.shape[0]):
                y_fit = right[0] * (i ** 2) + right[1] * i + right[2]
                cv2.circle(bird_right_col, (int(y_fit), i), 1, (0, 0, 255), thickness=1)


        cv2.imshow("Original", img)
        cv2.imshow("GT Left", left_img)
        cv2.imshow("GT Right", right_img)
        cv2.moveWindow("Original", 100, 100)
        cv2.moveWindow("GT Left", 800, 100)
        cv2.moveWindow("GT Right", 1500, 100)

        cv2.imshow("Original Bird", bird)
        cv2.imshow("GT Left Bird", bird_left_col)
        cv2.imshow("GT Right Bird", bird_right_col)
        cv2.moveWindow("Original Bird", 100, 800)
        cv2.moveWindow("GT Left Bird", 800, 800)
        cv2.moveWindow("GT Right Bird", 1500, 800)
        cv2.waitKey(1)

    print("=============== SUMMARY ============")
    print("Smaples: "+str(len(images)))
    print()
    print("mean_acc_left: "+str(np.mean(acc_left_list)))
    print("mean_acc_right: "+ str(np.mean(acc_right_list)))
    print("mean_acc: "+str(np.mean([acc_left_list, acc_right_list])))
    print()
    print("mean_deviation_left: " + str(np.mean(deviation_left_list)))
    print("mean_deviation_right: " + str(np.mean(deviation_right_list)))
    print("mean_deviation: " + str(np.mean([deviation_left_list, deviation_right_list])))
    print()
    print("mis_det_left: "+str(detection_left_list.count(False)/len(detection_left_list)))
    print("mis_det_right: "+str(detection_right_list.count(False)/len(detection_right_list)))
    mis_detection = [detection_left_list, detection_right_list]
    print("mis_det: "+str(mis_detection.count(False)/len(mis_detection)))


def main():

    images = [f for f in listdir(IMAGES) if isfile(join(IMAGES, f))]


    for path in images:

        width = cp.FRAME_WIDTH
        height = cp.FRAME_HEIGHT

        points = np.float32([
            [
                261,
                326
            ], [
                356,
                326
            ], [
                173,
                478
            ], [
                411,
                478
            ]
        ])
        destination_points = np.float32([
            [
                width / cp.CHESSBOARD_ROW_CORNERS,
                height / cp.CHESSBOARD_COL_CORNERS
            ], [
                width - (width / cp.CHESSBOARD_ROW_CORNERS),
                height / cp.CHESSBOARD_COL_CORNERS
            ], [
                width / cp.CHESSBOARD_ROW_CORNERS,
                height + 200
            ], [
                width - (width / cp.CHESSBOARD_ROW_CORNERS),
                height + 200
            ]
        ])

        M = cv2.getPerspectiveTransform(points, destination_points)

        birdview = BirdsEye(M=M, width=width, height=height)





        img = cv2.resize(img, (width, height))
        img = cv2.medianBlur(img, 3)  # remove noise from HS channels TODO choose

        # TODO adapt result to correct size

        bird = birdview.apply(img)

        # Line Detection
        left, right = ld.detect(bird, negate=False, robot=False, thin=False)
        print(left, right)

        cv2.imshow("Original", img)
        cv2.moveWindow("Original", 800, 100)
        cv2.waitKey(0)


if __name__ == '__main__':
    # convert_for_ground_true()
    evaluate()

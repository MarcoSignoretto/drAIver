import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from draiver.motion.motorcontroller import MotorController
from draiver.camera.birdseye import BirdsEye
import draiver.camera.properties as cp
from sklearn.preprocessing import normalize

HEIGHT = 480
WIDTH = 640

# BASE_PATH = "/mnt/B01EEC811EEC41C8/" # Ubuntu Config
BASE_PATH = "/Users/marco/Documents/" # Mac Config

INTERSECTION_LINE = 150

DEBUG = False
PLOT = False

MEDIAN_LINE_THRESHOLD = 30  #TODO  tested for robot
# WINDOW_LINE_THRESHOLD = 10
WINDOW_LINE_THRESHOLD = 5

def compute_hist(th2, pt1, pt2):
    """
    Compute the histogram for the portion of the image between the two points ( the current window )
    :param th2: binary thresholded image used to compute the histogram
    :param pt1: left_most, top_most point
    :param pt2: right_most, bottom_most point
    :return: tuple, histogram and it's origin
    """
    origin_x = max(pt1[0], 0)
    end_x = min(pt2[0], th2.shape[1])
    w_width = range(origin_x, end_x)

    hist = np.sum(th2[pt1[1]:pt2[1], origin_x:end_x], axis=0)

    if PLOT:
        plt.plot(w_width, hist)
        plt.show()

    if DEBUG:
        cv2.imshow("WINDOW", th2[pt1[1]:pt2[1], origin_x:end_x])
        cv2.moveWindow("WINDOW", 10, 10)
        cv2.waitKey(100)

    return hist, origin_x

def compute_window_line(hist, origin_x):
    """
    Extract the origin of the line given the histogram of a window
    :param hist:
    :param origin_x: offset of the histogram respect to the image dimension
    :return: the starting point of the line if there are enough evidence
    """
    line = None
      # TODO fix with better value
    max = np.argpartition(hist, -2)[-1:]

    if hist[max].squeeze() >= WINDOW_LINE_THRESHOLD:
        line = origin_x + max

    return line

def find_median_line(th2, mask):
    """

    :param th2: binary thresholded image used to compute the median line
    :return: x_values and y_values WARNING y is function of x where x is related to the height of the image
    """
    res = np.transpose(np.nonzero(th2))
    mask_res = np.transpose(np.nonzero(mask))
    np.sum(mask, axis=1) # TODO continue from here

    non_zero_indexes = np.nonzero(np.sum(mask, axis=1))

    x_values = []
    y_values = []
    for i in range(0, th2.shape[0]-1):
        if i in res[:, 0]:
            items = res[res[:, 0] == i, 1]
            items = [item for item in items if mask[i][item] == 255]

            if len(items) > MEDIAN_LINE_THRESHOLD:
                x_values.append(i)
                y_values.append(np.median(items))

    return x_values, y_values

def find_median_line_old(th2, from_x, to_x):
    """

    :param th2: binary thresholded image used to compute the median line
    :param from_x:
    :param to_x:
    :return: x_values and y_values WARNING y is function of x where x is related to the height of the image
    """
    res = np.transpose(np.nonzero(th2[:, from_x:to_x]))

    x_values = []
    y_values = []
    for i in range(0, th2.shape[0]-1):
        if i in res[:, 0]:
            items = res[res[:, 0] == i, 1]
            if len(items) > MEDIAN_LINE_THRESHOLD:
                x_values.append(i)
                y_values.append(from_x + np.median(items))

    return x_values, y_values

def compute_base_hist(th2):
    hist = np.sum(th2, axis=0)
    hist = np.divide(hist, np.repeat(np.max(hist), th2.shape[1]))
    return hist

def update_mask_for_line(th2, line, mask, window_width, window_height, debug_img = None):
    """
    Given the base line position (line) the function search to follow such line with a sliding window from bottom to top,
    at each sliding window step the mask will be updated, so in the end the obtained mask are related on where the lines
    points are located.
    Mask allows to perform a filter over all the thresholded image

    :param th2: binary thresholded image used to update the mask
    :param line: base line position
    :param mask: mask to update
    :param window_width: width of the window
    :param window_height: height of the window
    """
    height = th2.shape[0]
    margin = window_width / 2
    line_prev = line

    w_y = height
    while w_y >= window_height:

        origin_x = int(max(line_prev - margin, 0))
        end_x = int(min(line_prev + margin, th2.shape[1]))

        pt1 = (origin_x, int(w_y - window_height))
        pt2 = (end_x, int(w_y))
        mask[pt1[1]:pt2[1], pt1[0]:pt2[0]] = 255

        w_hist, w_origin = compute_hist(th2, pt1, pt2)
        maybe_line = compute_window_line(w_hist, w_origin)

        # If no line has been detected on window keep the center of the previous line
        if maybe_line is not None:
            line_prev = maybe_line

        w_y = w_y - window_height

        if DEBUG:
            cv2.rectangle(debug_img, pt1, pt2, (0, 255, 0), thickness=3, lineType=cv2.LINE_8)


def detect(img, negate=False, robot=False):
    left = None
    right = None

    width = img.shape[1]
    height = img.shape[0]

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    left_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    right_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    if not negate:
        gray = np.zeros((height, width, 1), dtype=np.uint8)
    else:
        gray = np.full((height, width, 1), 255, dtype=np.uint8)
    cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, gray, 1)
    if negate:
        gray = abs(255 - gray)

    # V2 implementation use equalization of histogram
    # FIXME bad on robot
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray = clahe.apply(gray)

    # Use adaptive thresholding because it is better for difficult lighting condition
    if robot:
        th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, -15)  # TODO values for ROBOT   this should be ok
    else:
        th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, -35)  #TODO values for KITTY   maybe use a bit little biass

    if DEBUG:
        cv2.imshow("No erosion", th2)
        cv2.moveWindow("No erosion", 800, 600)
    th2 = cv2.erode(th2, kernel=(7, 7, 1), iterations=3)  # TODO test

    #th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #TODO values for KITTY   maybe use a bit little biass

    # ===============================  Calculate base histogram =============================

    hist = compute_base_hist(th2)
    if PLOT:
        plt.plot(range(0, th2.shape[1]), hist)
        plt.show()

    # get highest values
    half_width = (width/2)
    left_max = np.argpartition(hist[0:int(half_width-1)], -2)[-1:]
    right_max = np.argpartition(hist[int(half_width):int(width-1)], -2)[-1:] + int(half_width)

    print(left_max, right_max)

    # =============================== THRESHOLD ========================================
    LINE_THRESHOLD = 0.3
    left_line = None
    right_line = None

    WINDOW_WIDTH = 100
    WINDOW_HEIGHT = height / 6

    if hist[left_max].squeeze() >= LINE_THRESHOLD:
        left_line = left_max
    if hist[right_max].squeeze() >= LINE_THRESHOLD:
        right_line = right_max

    # # ======================= LEFT LINE
    if left_line is not None:
        update_mask_for_line(th2, left_line, left_mask, WINDOW_WIDTH, WINDOW_HEIGHT, debug_img=img)
    else:
        print("LEFT LINE NONE!!!")

    # # ======================= RIGHT LINE
    if right_line is not None:
        update_mask_for_line(th2, right_line, right_mask, WINDOW_WIDTH, WINDOW_HEIGHT, debug_img=img)
    else:
        print("RIGHT LINE NONE!!!")

    cv2.imshow("th2m", th2)
    cv2.moveWindow("th2m", 10, 700)

    # ================================ MASKING REGIONS ================================

    mask = cv2.bitwise_or(mask, left_mask)
    mask = cv2.bitwise_or(mask, right_mask)

    # if DEBUG:
    #     cv2.imshow("Mask", mask)
    #     cv2.moveWindow("Mask", 200, 10)
    #
    th2 = cv2.bitwise_and(th2, mask)

    if DEBUG:
        cv2.imshow("th2m", th2)
        cv2.moveWindow("th2m", 10, 700)

        cv2.imshow("left_mask", left_mask)
        cv2.moveWindow("left_mask", 10, 700)
        cv2.imshow("right_mask", right_mask)
        cv2.moveWindow("right_mask", 650, 700)

        cv2.imshow("mask", mask)
        cv2.moveWindow("mask", 1500, 700)

    # ================================ POLYNOMIAL FIT ================================

    # TODO FIX ME
    #  x_values_left, y_values_left = find_median_line(th2, from_x=0, to_x=int((width/2)-1))
    #  x_values_right, y_values_right = find_median_line(th2, from_x=int(width/2), to_x=int(width-1))

    x_values_left, y_values_left = find_median_line(th2, mask=left_mask)
    x_values_right, y_values_right = find_median_line(th2, mask=right_mask)

    if DEBUG:
        for i in range(0, len(x_values_left)-1):
            cv2.circle(img, (int(y_values_left[i]), int(x_values_left[i])), 1, (255, 0, 0), thickness=1)
        for i in range(0, len(x_values_right) - 1):
            cv2.circle(img, (int(y_values_right[i]), int(x_values_right[i])), 1, (255, 0, 0), thickness=1)

    if len(x_values_left) > 10: # TODO fix custom threshold for realiable line
        left_fit = np.polyfit(x_values_left, y_values_left, 2)
        # TODO check residuals for quality
        left = left_fit

        if DEBUG:
            for i in range(0, img.shape[0]-1):
                y_fit = left_fit[0]*(i**2) + left_fit[1]*i + left_fit[2]
                cv2.circle(img, (int(y_fit), i), 1, (0, 0, 255), thickness=1)

    if len(x_values_right) > 10:  # TODO fix custom threshold for realiable line
        right_fit = np.polyfit(x_values_right, y_values_right, 2)
        # TODO check residuals for quality
        right = right_fit

        if DEBUG:
            for i in range(0, img.shape[0] - 1):
                y_fit = right_fit[0] * (i ** 2) + right_fit[1] * i + right_fit[2]
                cv2.circle(img, (int(y_fit), i), 1, (0, 0, 255), thickness=1)

    if DEBUG:
        pass
        # cv2.circle(img, (int(np.round(left)), img.shape[0] - INTERSECTION_LINE), 5, (0, 0, 255), thickness=2)
        # cv2.circle(img, (int(np.round(right)), img.shape[0] - INTERSECTION_LINE), 5, (0, 0, 255), thickness=2)
        #
        # #  center
        # lines_range = right - left
        # mid = left + lines_range / 2
        # cv2.circle(img, (int(np.round(mid)), img.shape[0] - INTERSECTION_LINE), 7, (0, 255, 255), thickness=2)
        #
        # #  car position
        # cv2.circle(img, (int(np.round(car_position)), img.shape[0] - INTERSECTION_LINE), 5, (14, 34, 255), thickness=5)
        #
        # #cv2.imshow("Gray", gray)
        # cv2.imshow("Otzu", thr)

        # cv2.imshow("Img", img)
        # cv2.imshow("Adapt gaussian", th3)
        # cv2.imshow("Canny", edges)
        # cv2.imshow("CannyDilated", dilate)
        # cv2.imshow("Adapt mean erosion", th2erosion)
        # cv2.imshow("Adapt gaussian erosion", th3erosion)
        # cv2.imshow("erosion", erosion)

    cv2.imshow("Adapt mean", th2)
    cv2.moveWindow("Adapt mean", 1500, 100)
    cv2.waitKey(1)

    return left, right

# TODO remove
def test_color_space():
    imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # imgHSV[] = cv2.equalizeHist(V)

    # TODO cut image on half to have strongest line detection and avoid noise

    # TODO fix linee tratteggiate

    L, A, B = cv2.split(imgLAB)
    H, S, V = cv2.split(imgHSV)
    h, l, s = cv2.split(imgHLS)

    # =========================

    # python
    bgr = [215, 215, 215]
    thresh = 45

    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

    maskBGR = cv2.inRange(img, minBGR, maxBGR)
    resultBGR = cv2.bitwise_and(img, img, mask=maskBGR)

    # ======================================= HSV ==================================
    # convert 1D array to 3D, then convert it to HSV and take the first element
    # this will be same as shown in the above figure [65, 229, 158]
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    hsvMIN = [0, 0, 200]
    hsvMAX = [50, 20, 255]

    # minHSV = np.array([hsvMIN[0], hsvMIN[1], hsvMIN[2]])
    # maxHSV = np.array([hsvMAX[0], hsvMAX[1], hsvMAX[2]])

    minHSV = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
    maxHSV = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])

    maskHSV = cv2.inRange(imgHSV, minHSV, maxHSV)
    resultHSV = cv2.bitwise_and(imgHSV, imgHSV, mask=maskHSV)
    resultHSV = cv2.cvtColor(resultHSV, cv2.COLOR_HSV2BGR)

    # ======================================= LAB ==================================
    # convert 1D array to 3D, then convert it to LAB and take the first element
    lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]

    minLAB = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
    maxLAB = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])

    maskLAB = cv2.inRange(imgLAB, minLAB, maxLAB)
    resultLAB = cv2.bitwise_and(imgLAB, imgLAB, mask=maskLAB)
    resultLAB = cv2.cvtColor(resultLAB, cv2.COLOR_LAB2BGR)

    # ======================================= HLS ==================================

    # AUTO
    hls = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HLS)[0][0]

    minHLS = np.array([hls[0] - thresh, hls[1] - thresh, hls[2] - thresh])
    maxHLS = np.array([hls[0] + thresh, hls[1] + thresh, hls[2] + thresh])
    maskHLS = cv2.inRange(imgHLS, minHLS, maxHLS)

    resultHLS = cv2.bitwise_and(imgHLS, imgHLS, mask=maskHLS)
    resultHLS = cv2.cvtColor(resultHLS, cv2.COLOR_HLS2BGR)

    # #with different setting
    #
    # #hls = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HLS)[0][0]
    # h, l, s = cv2.split(imgHLS)
    # hlsMIN = [0, 0, 0]
    # hlsMAX = [110, 255, 45]
    #
    # # use full range for l because then thresholded
    # l = cv2.adaptiveThreshold(l, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31,
    #                           -10)  # maybe use a bit little biass
    #
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # s = clahe.apply(s)
    #
    # # s = cv2.equalizeHist(s)
    #
    #
    #
    # # # ================ hist norm ====================
    # # hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    # # cdf = hist.cumsum()
    # # cdf_normalized = cdf * hist.max() / cdf.max()
    # # plt.plot(cdf_normalized, color='b')
    # # plt.hist(img.flatten(), 256, [0, 256], color='r')
    # # plt.xlim([0, 256])
    # # plt.legend(('cdf', 'histogram'), loc='upper left')
    # # plt.show()
    #
    #
    # # minHLS = np.array([hls[0] - thresh, hls[1] - thresh, hls[2] - thresh])
    # # maxHLS = np.array([hls[0] + thresh, hls[1] + thresh, hls[2] + thresh])
    # minHLS = np.array([hlsMIN[0], hlsMIN[1], hlsMIN[2]])
    # maxHLS = np.array([hlsMAX[0], hlsMAX[1], hlsMAX[2]])
    #
    #
    # maskHLS = cv2.inRange(imgHLS, minHLS, maxHLS)
    #
    # resultHLS = cv2.bitwise_and(imgHLS, imgHLS, mask=maskHLS)
    # #
    # resultHLS = cv2.bitwise_and(resultHLS, resultHLS, mask=l)
    #
    # resultHLS = cv2.cvtColor(resultHLS, cv2.COLOR_HLS2BGR)

    # ========================


if __name__ == '__main__':
    # Adaptive gaussian or mean ( gaussian is a bit better )
    # gaussian a difficoltà sul molto scuro

    images = [
        # "Datasets/drAIver/line_detection/street.jpg",
        "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000009.png",
        "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000014.png",
        "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000024.png",
        "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000044.png",
        "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000047.png",
        "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000071.png",
        "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000123.png",
        # linee trateggiate
        "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000081.png",
        "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000087.png"
    ]

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
                height
            ], [
                width - (width / cp.CHESSBOARD_ROW_CORNERS),
                height
            ]
        ])

        M = cv2.getPerspectiveTransform(points, destination_points)

        birdeye = BirdsEye(M=M, width=width, height=height)

        img = cv2.imread(BASE_PATH + path)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        #img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.medianBlur(img, 3)  # remove noise from HS channels TODO choose

        # Work only on bird view
        if DEBUG:
            cv2.imshow("Frame", img)
        img = birdeye.apply(img)

        # ======================== DETECTION ===========================

        left, right = detect(img)

        # ======================== PLOT ===========================

        if left is not None:
            for i in range(0, img.shape[0]-1):
                y_fit = left[0]*(i**2) + left[1]*i + left[2]
                cv2.circle(img, (int(y_fit), i), 1, (0, 0, 255), thickness=1)

        if right is not None:
            for i in range(0, img.shape[0] - 1):
                y_fit = right[0] * (i ** 2) + right[1] * i + right[2]
                cv2.circle(img, (int(y_fit), i), 1, (0, 0, 255), thickness=1)

        cv2.imshow("Frame d", img)
        cv2.moveWindow("Frame d", 100, 100)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

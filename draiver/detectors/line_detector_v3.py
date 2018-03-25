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
from numpy.polynomial import polynomial as P

HEIGHT = 480
WIDTH = 640

BASE_PATH = "/mnt/B01EEC811EEC41C8/" # Ubuntu Config
#BASE_PATH = "/Users/marco/Documents/" # Mac Config

INTERSECTION_LINE = 150

DEBUG = True
PLOT = False

#  =========================== TESTS ==============================


def rho_theta_test():
    x1 = 100
    y1 = 200

    x2 = 100
    y2 = 155

    img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000009.png")

    img = cv2.resize(img, (WIDTH, HEIGHT))

    # detect(img)

    # ================= POINT ============================

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=3, lineType=cv2.LINE_8)

    # ============== RHO THETA ==========================

    theta = compute_theta(x1, y1, x2, y2)
    rho = compute_rho(x1, y1, x2, y2)

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2, lineType=cv2.LINE_8)

    cv2.imshow("frame", img)


def intersection_test():
    #  =================== TEST ========================
    q = 200
    m = 1.5
    # m = int(np.round(np.tan(theta)))
    # m = int(np.round(np.tan(0.78)))

    x1 = 0
    y1 = 0 + q
    #
    x2 = 480
    y2 = int(np.round(m * x2 + q))

    cv2.line(img, (x1, y1), (x2, y2), (128, 255, 23), thickness=3, lineType=cv2.LINE_8)

    #  =================== TEST POLAR PLOT ========================

    theta = compute_theta(x1, y1, x2, y2)
    rho = compute_rho(x1, y1, x2, y2)
    print("theta:" + str(theta))
    print("rho:" + str(rho))

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    pt1 = (int(round(x0 + 1000 * (-b))), int(round(y0 + 1000 * (a))))
    # pt1 = (int(round(y0 + 1000 * (a))), int(round(x0 + 1000 * (-b))))
    pt2 = (int(round(x0 - 1000 * (-b))), int(round(y0 - 1000 * (a))))
    # pt2 = (int(round(y0 - 1000 * (a))), int(round(x0 - 1000 * (-b))))

    cv2.line(img, pt1, pt2, (128, 55, 23), thickness=1, lineType=cv2.LINE_8)

    #  =================== TEST POLAR ========================

    K = img.shape[0] - INTERSECTION_LINE

    q = compute_q(rho, theta)
    # q = rho * np.sqrt(1+np.power(np.tan(theta), 2))
    x_new = (K - q) / compute_m_from_polar(theta)

    # q_new = K - np.sin(theta) * rho
    # x_new = (K-q_new)/m

    print("x_new:" + str(x_new) + " q_new: " + str(q))

    cv2.circle(img, (int(np.round(rho * np.cos(theta))), int(np.round(rho * np.sin(theta)))), 5, (255, 34, 100),
               thickness=1)

    cv2.circle(img, (int(np.round(x_new)), K), 5, (134, 234, 100), thickness=2)

    cv2.imshow("Img", img)


def compute_theta(x1, y1, x2, y2):
    if x1-x2 == 0:
        return 0
    else:
        return np.arctan((y1 - y2) / (x1 - x2))+(np.pi/2.0)


def compute_m(x1, y1, x2, y2):

    return (y1 - y2) / (x1 - x2)


def compute_rho(x1, y1, x2, y2):
    if x1-x2 == 0:
        return x1 # distance from origin
    else:
        m = compute_m(x1, y1, x2, y2)
        q = y1-m*x1

        return abs(q)/np.sqrt(1.0+math.pow(m, 2))


def compute_m_from_polar(theta):
    if np.sin(theta) != 0:
        return -np.cos(theta) / np.sin(theta)
    else:
        return math.nan


def compute_q(rho, theta):
    if np.sin(theta) != 0:
        m = -np.cos(theta) / np.sin(theta)
        q = rho * np.sqrt(1 + np.power(m, 2))
        return q
    else:
        return math.nan


def cluster_lines(line_points):
    line_points = StandardScaler().fit_transform(line_points)

    if PLOT:
        colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
        colors = np.hstack([colors] * 20)

    # Compute DBSCAN
    db = DBSCAN(eps=0.1, min_samples=1).fit(line_points)
    if PLOT:
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(X, labels))

    # #############################################################################

    # if hasattr(db, 'labels_'):
    y_pred = db.labels_.astype(np.int)
    # else:
    #    y_pred = db.predict(line_points)

    if PLOT:
        for i in range(0, len(line_points)):
            plt.scatter(line_points[i][0], line_points[i][1], color=colors[y_pred[i]].tolist(), s=10)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
    return db


def kmeans(line_points):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(line_points)
    # >> > kmeans.labels_
    # array([0, 0, 0, 1, 1, 1], dtype=int32)
    # >> > kmeans.predict([[0, 0], [4, 4]])
    # array([0, 1], dtype=int32)
    # >> > kmeans.cluster_centers_
    # array([[1., 2.],
    #        [4., 2.]])

    for p, l in zip(line_points, kmeans.labels_):
        if l == 1:
            color = "blue"
        else:
            color = "red"
        plt.scatter(p[0], p[1], color=color)

    for line in kmeans.cluster_centers_:
        plt.scatter(line[0], line[1], color="green")

        a = math.cos(line[1])
        b = math.sin(line[1])

        x0 = a * line[0]
        y0 = b * line[0]

        pt1 = (int(round(x0 + 1000 * (-b))), int(round(y0 + 1000 * (a))))
        # pt1 = (int(round(y0 + 1000 * (a))), int(round(x0 + 1000 * (-b))))
        pt2 = (int(round(x0 - 1000 * (-b))), int(round(y0 - 1000 * (a))))
        # pt2 = (int(round(y0 - 1000 * (a))), int(round(x0 - 1000 * (-b))))

        cv2.line(img, pt1, pt2, (128, 55, 23), thickness=3, lineType=cv2.LINE_8)


def find_intersections(lines, reference):

    intersection_x = []

    for line in lines:
        theta = line[1]
        rho = line[0]
        if np.sin(theta) != 0:
            m = compute_m_from_polar(theta)
            q = compute_q(rho, theta)
            x = (reference - q) / m
        else:
            x = rho #TODO test

        intersection_x.append(x)

    return intersection_x


def filter_road_lines(car_position, intersections, image_width):
    left = 0
    right = image_width

    for intersection in intersections:
        if intersection < car_position and left < intersection:
            left = intersection
        elif intersection > car_position and right > intersection:
            right = intersection

    return left, right


def compute_hist(th2, pt1, pt2):
    """

    :param th2:
    :param pt1:
    :param pt2:
    :return: tuple, histogram and it's origin
    """
    origin_x = max(pt1[0], 0)
    end_x = min(pt2[0], th2.shape[1])
    w_width = range(origin_x, end_x)

    cv2.imshow("WINDOW", th2[pt1[1]:pt2[1], origin_x:end_x])
    cv2.moveWindow("WINDOW", 10, 10)

    hist = np.sum(th2[pt1[1]:pt2[1], origin_x:end_x], axis=0)
    ##hist = cv2.normalizeHist(hist, np.max(hist))
    #hist = np.divide(hist, np.repeat(np.max(hist), w_width))
    plt.plot(w_width, hist)
    plt.show()

    cv2.waitKey(100)

    return hist, origin_x

def compute_window_line(hist, origin_x):
    line = None
    WINDOW_LINE_THRESHOLD = 10
    max = np.argpartition(hist, -2)[-1:]

    if hist[max].squeeze() >= WINDOW_LINE_THRESHOLD:
        line = origin_x + max

    return line


def detect(img, negate = False):

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    gray = np.zeros((HEIGHT, WIDTH, 1), dtype=np.uint8)
    cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, gray, 1)
    if negate:
        gray = abs(255 - gray)

    # V2 implementation use equalization of histogram
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, -35)  # maybe use a bit little biass

    # ===============================  Calculate base histogram =============================

    # gray_norm = normalize(gray, axis=1)
    hist = np.sum(th2, axis=0)
    ##hist = cv2.normalizeHist(hist, np.max(hist))
    hist = np.divide(hist, np.repeat(np.max(hist), cp.FRAME_WIDTH))
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
    margin = WINDOW_WIDTH / 2

    if hist[left_max].squeeze() >= LINE_THRESHOLD:
        left_line = left_max
    if hist[right_max].squeeze() >= LINE_THRESHOLD:
        right_line = right_max

    # ======================= LEFT LINE
    if left_line is not None:
        pt1 = (int(left_line - margin), int(height - WINDOW_HEIGHT))
        pt2 = (int(left_line + margin), height)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=3, lineType=cv2.LINE_8)
        mask[pt1[1]:pt2[1], pt1[0]:pt2[0]] = 255
        left_prev = left_line
        w_y = height - WINDOW_HEIGHT
        for i in range(0, 5):
            pt1 = (int(left_prev - margin), int(w_y - WINDOW_HEIGHT))
            pt2 = (int(left_prev + margin), int(w_y))
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=3, lineType=cv2.LINE_8)
            mask[pt1[1]:pt2[1], pt1[0]:pt2[0]] = 255



            w_hist, w_origin = compute_hist(th2, pt1, pt2)
            maybe_left = compute_window_line(w_hist, w_origin)

            if maybe_left is not None:
                left_prev = maybe_left

            w_y = w_y - WINDOW_HEIGHT
    else:
        print("LEFT LINE NONE!!!")

    # ======================= RIGHT LINE
    if right_line is not None:
        pt1 = (int(right_line - margin), int(height - WINDOW_HEIGHT))
        pt2 = (int(right_line + margin), height)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=3, lineType=cv2.LINE_8)
        mask[pt1[1]:pt2[1], pt1[0]:pt2[0]] = 255
        right_prev = right_line
        w_y = height - WINDOW_HEIGHT
        for i in range(0, 5):
            pt1 = (int(right_prev - margin), int(w_y - WINDOW_HEIGHT))
            pt2 = (int(right_prev + margin), int(w_y))
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=3, lineType=cv2.LINE_8)

            w_hist, w_origin = compute_hist(th2, pt1, pt2)
            maybe_right = compute_window_line(w_hist, w_origin)
            mask[pt1[1]:pt2[1], pt1[0]:pt2[0]] = 255
            
            if maybe_right is not None:
                right_prev = maybe_right

            w_y = w_y - WINDOW_HEIGHT
    else:
        print("RIGHT LINE NONE!!!")

    # ================================ MASKING REGIONS ================================

    cv2.imshow("Mask", mask)
    cv2.moveWindow("Mask", 200, 10)

    th2 = cv2.bitwise_and(th2, mask)

    cv2.imshow("th2m", th2)
    cv2.moveWindow("th2m", 10, 700)

    # ================================ POLYNOMIAL FIT ================================

    # TODO continuare da qui
    left_samples = np.median(th2[:, 0:int(width/2)], axis=1) # FIXME it not works
    # dividere immagine in 2
    # fare mediana rispetto a x in modo da estrarre 1 linea
    # poi usare poly fit

    P.polyfit()

    lines = cv2.HoughLines(th2, 1, np.pi/180, 200)
    filtered_lines = []
    for item in lines:

        rho, theta = item[0]

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2, lineType=cv2.LINE_8)

        if theta < 0.78 or theta > 2.35: #TODO fix theta
            if DEBUG:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1, lineType=cv2.LINE_8)

            filtered_lines.append((rho, theta))

    print("============ Filtered lines =============")

    rhos = []
    thetas = []
    for rho, theta in filtered_lines:
        thetas.append(theta)
        rhos.append(rho)

    if PLOT:
        plt.scatter(rhos, thetas)
        plt.show()

    line_points = [list(t) for t in zip(rhos, thetas)]

    # ===== CLUSTERING ============

    cluster_result = cluster_lines(line_points)

    labels = cluster_result.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_items = {}
    for i in range(0, n_clusters):
        cluster_items[i] = []

    for i in range(0, len(labels)):
        cluster = labels[i]
        if cluster != -1:
            cluster_items[cluster].append(np.asarray(line_points[i]))

    centroids = []
    for key, value in cluster_items.items():
        centroids.append(np.asarray(value).mean(axis=0))

    for center in centroids:
        a = math.cos(center[1])
        b = math.sin(center[1])

        x0 = a * center[0]
        y0 = b * center[0]

        pt1 = (int(round(x0 + 1000 * (-b))), int(round(y0 + 1000 * (a))))
        # pt1 = (int(round(y0 + 1000 * (a))), int(round(x0 + 1000 * (-b))))
        pt2 = (int(round(x0 - 1000 * (-b))), int(round(y0 - 1000 * (a))))
        # pt2 = (int(round(y0 - 1000 * (a))), int(round(x0 - 1000 * (-b))))

        if DEBUG:
            cv2.line(img, pt1, pt2, (0, 0, 0), thickness=3, lineType=cv2.LINE_8)

    if PLOT:
        plt.show()

    pt1 = (0, img.shape[0]-INTERSECTION_LINE)
    pt2 = (img.shape[1], img.shape[0]-INTERSECTION_LINE)

    if DEBUG:
        cv2.line(img, pt1, pt2, (34, 112, 200), thickness=4, lineType=cv2.LINE_8)

    #==================== CALCULATE INTERSECTIONS ==========================

    intersections = find_intersections(centroids, img.shape[0]-INTERSECTION_LINE)

    for intersection in intersections:
        if intersection >= 0 and intersection <= img.shape[1]:
            if DEBUG:
                cv2.circle(img, (int(np.round(intersection)), img.shape[0]-INTERSECTION_LINE), 5, (134, 234, 100), thickness=2)

    #==================== FIND 2 ROAD LINES ========================

    car_position = img.shape[1] / 2

    left, right = filter_road_lines(car_position, intersections, img.shape[1])

    if DEBUG:
        cv2.circle(img, (int(np.round(left)), img.shape[0] - INTERSECTION_LINE), 5, (0, 0, 255), thickness=2)
        cv2.circle(img, (int(np.round(right)), img.shape[0] - INTERSECTION_LINE), 5, (0, 0, 255), thickness=2)

        #  center
        lines_range = right - left
        mid = left + lines_range / 2
        cv2.circle(img, (int(np.round(mid)), img.shape[0] - INTERSECTION_LINE), 7, (0, 255, 255), thickness=2)

        #  car position
        cv2.circle(img, (int(np.round(car_position)), img.shape[0] - INTERSECTION_LINE), 5, (14, 34, 255), thickness=5)

        #cv2.imshow("Gray", gray)
        # cv2.imshow("Otzu", thr)

        cv2.imshow("Adapt mean", th2)
        cv2.moveWindow("Adapt mean", 1500, 100)
        #cv2.imshow("Img", img)
        # cv2.imshow("Adapt gaussian", th3)
        # cv2.imshow("Canny", edges)
        # cv2.imshow("CannyDilated", dilate)
        # cv2.imshow("Adapt mean erosion", th2erosion)
        # cv2.imshow("Adapt gaussian erosion", th3erosion)
        # cv2.imshow("erosion", erosion)

        cv2.waitKey(1)

    return left, right, car_position

# Morphology filter
def morphology_filter(img_):
    """

    :param img_: Input Image
    :return: Morphologically Filtered Image
    """
    gray = np.zeros((HEIGHT, WIDTH, 1), dtype=np.uint8)
    gray = cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY)
    hls_s = cv2.cvtColor(img_, cv2.COLOR_RGB2HLS)[:, :, 2]
    src = 0.6*hls_s + 0.4*gray
    src = np.array(src-np.min(src)/(np.max(src)-np.min(src))).astype('float32')-0.5
    blurf = np.zeros((1, 5))
    blurf.fill(1)
    src = cv2.filter2D(src, cv2.CV_32F, blurf)
    f = np.zeros((1, 30))
    f.fill(1)
    l = cv2.morphologyEx(src, cv2.MORPH_OPEN, f)
    filtered = src - l
    return filtered


# Image Threshold
def img_threshold(img_):
    """

    :param img_: Input Image
    :return: Thresholded Image
    """

    # Use HLS, LAB Channels for Thresholding #
    img_hls = cv2.cvtColor(img_, cv2.COLOR_RGB2HLS)
    img_hls = cv2.medianBlur(img_hls, 5)
    b_channel = cv2.cvtColor(img_, cv2.COLOR_RGB2LAB)[:, :, 2]
    b_channel = cv2.medianBlur(b_channel, 5)

    # Filter out Greenery & Soil from environment
    environment = np.logical_not(
        (b_channel > 145) & (b_channel < 200) & cv2.inRange(img_hls, (0, 0, 50), (35, 192, 255))).astype(np.uint8) & (
                              img_hls[:, :, 1] < 245)

    # Deal with shadows and bright spots on the road #
    # The shapes can be elliptical or rectangular #
    big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    environment = cv2.morphologyEx(environment, cv2.MORPH_OPEN, small_kernel)
    environment_mask = cv2.dilate(environment, big_kernel)

    # Use Lane from Last mask to filter ROI
    img_mask = environment_mask  #cv2.bitwise_and(last_mask, environment_mask)

    # Morphology Channel Thresholding
    morph_channel = morphology_filter(img_)
    morph_thresh_lower = 1.2 * np.mean(morph_channel) + 1.3 * np.std(morph_channel)
    morph_thresh_upper = np.max(morph_channel)
    morph_binary = np.zeros_like(morph_channel)
    morph_binary[(morph_channel >= morph_thresh_lower) &
                 (morph_channel <= morph_thresh_upper)] = 1

    # Erosion Kernel to clear out small granular noises
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_binary = cv2.morphologyEx(morph_binary.astype(np.uint8), cv2.MORPH_ERODE, erosion_kernel)
    morph_binary = cv2.morphologyEx(morph_binary, cv2.MORPH_OPEN, small_kernel)
    combined_binary = cv2.bitwise_and(morph_binary.astype(np.uint8), img_mask.astype(np.uint8))

    return combined_binary.astype(np.uint8)


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
        cv2.imshow("Frame", img)
        img = birdeye.apply(img)



        imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        #imgHSV[] = cv2.equalizeHist(V)



        # TODO cut image on half to have strongest line detection and avoid noise

        # TODO fix linee tratteggiate



        L, A, B = cv2.split(imgLAB)
        H, S, V = cv2.split(imgHSV)
        h, l, s = cv2.split(imgHLS)





        #=========================





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


        #minHSV = np.array([hsvMIN[0], hsvMIN[1], hsvMIN[2]])
        #maxHSV = np.array([hsvMAX[0], hsvMAX[1], hsvMAX[2]])

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

        #========================
        left, right, car_position = detect(img)

        #l = cv2.adaptiveThreshold(l, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -10)  # maybe use a bit little biass



        # cv2.imshow("L", L)
        # cv2.imshow("A", A)
        # cv2.imshow("B", B)

        # cv2.imshow("H", H)
        # cv2.moveWindow("H", 100, 100)
        # cv2.imshow("S", S)
        # cv2.moveWindow("S", 800, 100)
        # cv2.imshow("V", V)
        # cv2.moveWindow("V", 1500, 100)
        #
        # cv2.imshow("h", h)
        # cv2.moveWindow("h", 100, 700)
        # cv2.imshow("l", l)
        # cv2.moveWindow("l", 800, 700)
        # cv2.imshow("s", s)
        # cv2.moveWindow("s", 1500, 700)

        # cv2.imshow("Result HSV", gray)
        # cv2.moveWindow("Result HSV", 800, 100)


        # cv2.imshow("Result HSV", resultHSV)
        # cv2.moveWindow("Result HSV", 1500, 100)
        #
        # cv2.imshow("Result HLS", resultHLS)
        # cv2.moveWindow("Result HLS", 100, 800)

        # Work only on bird view
        cv2.imshow("Frame d", img)
        cv2.moveWindow("Frame d", 100, 100)

        # cv2.imshow("My", th2)
        # cv2.moveWindow("My", 800, 800)
        #
        # cv2.imshow("Gray", gray)
        # cv2.moveWindow("Gray", 1500, 800)

        # img = img_threshold(img)
        #
        # # MOrph
        # cv2.imshow("Morph", img)
        # cv2.moveWindow("Morph", 100, 700)



        cv2.waitKey(0)


    cv2.destroyAllWindows()

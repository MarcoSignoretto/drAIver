import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from draiver.motion.motorcontroller import MotorController
import draiver.camera.properties as cp
from draiver.camera.birdseye import BirdsEye

HEIGHT = 480
WIDTH = 640

# BASE_PATH = "/mnt/B01EEC811EEC41C8/"
BASE_PATH = "/Users/marco/Documents/" # Mac Config

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


def detect(img, negate = False):

    gray = np.zeros((HEIGHT, WIDTH, 1), dtype=np.uint8)
    cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, gray, 1)
    if negate:
        gray = abs(255 - gray)

    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -35)  # maybe use a bit little biass

    lines = cv2.HoughLines(th2, 1, np.pi/180, 200)
    filtered_lines = []
    if lines is not None:
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

            cv2.imshow("Gray", gray)
            # cv2.imshow("Otzu", thr)
            cv2.imshow("Adapt mean", th2)
            cv2.moveWindow("Adapt mean", 100, 800)
            #cv2.imshow("Img", img)
            # cv2.imshow("Adapt gaussian", th3)
            # cv2.imshow("Canny", edges)
            # cv2.imshow("CannyDilated", dilate)
            # cv2.imshow("Adapt mean erosion", th2erosion)
            # cv2.imshow("Adapt gaussian erosion", th3erosion)
            # cv2.imshow("erosion", erosion)

            cv2.waitKey(1)

        return left, right, car_position
    else:
        return None, None, None


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
        # points = np.float32([
        #     [
        #         281,
        #         266
        #     ], [
        #         326,
        #         263
        #     ], [
        #         159,
        #         479
        #     ], [
        #         420,
        #         478
        #     ]
        # ])
        # Fixed coordinate for road view
        destination_points = np.float32([
            [
                width / cp.CHESSBOARD_ROW_CORNERS,
                height / cp.CHESSBOARD_COL_CORNERS
            ], [
                width - (width / cp.CHESSBOARD_ROW_CORNERS),
                height / cp.CHESSBOARD_COL_CORNERS
            ], [
                width / cp.CHESSBOARD_ROW_CORNERS,
                height - (height / cp.CHESSBOARD_COL_CORNERS)
            ], [
                width - (width / cp.CHESSBOARD_ROW_CORNERS),
                height - (height / cp.CHESSBOARD_COL_CORNERS)
            ]
        ])

        M = cv2.getPerspectiveTransform(points, destination_points)

        birdeye = BirdsEye(M, width, height)

        img = cv2.imread(BASE_PATH + path)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        # img = birdeye.apply(img)

        # TODO cut image on half to have strongest line detection and avoid noise

        # TODO fix linee tratteggiate

        left, right, car_position = detect(img)

        # motor_controller = MotorController()
        # motor_controller.start()
        # motor_controller.get_queue().put((left, right, car_position))

        cv2.imshow("Frame", img)

        cv2.waitKey(0)
    cv2.destroyAllWindows()

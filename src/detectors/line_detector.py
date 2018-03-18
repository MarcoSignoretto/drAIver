import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

HEIGHT = 480
WIDTH = 640

BASE_PATH = "/mnt/B01EEC811EEC41C8/"

INTERSECTION_LINE = 70


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


def cluster_lines(line_points):
    line_points = StandardScaler().fit_transform(line_points)

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    # Compute DBSCAN
    db = DBSCAN(eps=0.1, min_samples=1).fit(line_points)
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

    for i in range(0, len(line_points)):
        plt.scatter(line_points[i][0], line_points[i][1], color=colors[y_pred[i]].tolist(), s=10)

    # # Plot result
    #
    # # Black removed and is used for noise instead.
    # unique_labels = set(labels)
    # colors = [plt.cm.Spectral(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = (labels == k)
    #
    #     xy = line_points[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=14)
    #
    #     xy = line_points[class_member_mask & ~core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=6)

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
        if np.cos(theta) != 0:
            m = np.tan(theta)
            q = reference - np.sin(theta) * rho
            x = (reference - q) / m
        else:
            x = rho #TODO test

        intersection_x.append(x)

    return intersection_x


def detect(img, negate = False):

    gray = np.zeros((HEIGHT, WIDTH, 1), dtype=np.uint8)
    cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, gray, 1)
    if negate:
        gray = abs(255 - gray)

    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -35)  # maybe use a bit little biass

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
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1, lineType=cv2.LINE_8)

            filtered_lines.append((rho,theta))
        else:
            print("Theta => "+str(theta))

    print("============ Filtered lines =============")

    rhos = []
    thetas = []
    for rho, theta in filtered_lines:
        thetas.append(theta)
        rhos.append(rho)

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

        cv2.line(img, pt1, pt2, (0, 0, 0), thickness=3, lineType=cv2.LINE_8)

    plt.show()

    pt1 = (0, img.shape[0]-INTERSECTION_LINE)
    pt2 = (img.shape[1], img.shape[0]-INTERSECTION_LINE)
    cv2.line(img, pt1, pt2, (34, 112, 200), thickness=4, lineType=cv2.LINE_8)

    #==================== CALCULATE INTERSECTIONS ==========================

    intersections = find_intersections(centroids, img.shape[0]-INTERSECTION_LINE)

    for intersection in intersections:
        if intersection >= 0 and intersection <= img.shape[1]:
            cv2.circle(img, (int(np.round(intersection)), img.shape[0]-INTERSECTION_LINE), 5, (134, 234, 100), thickness=2)

    cv2.imshow("Img", img)
    cv2.imshow("Gray", gray)
    #cv2.imshow("Otzu", thr)
    cv2.imshow("Adapt mean", th2)
    #cv2.imshow("Adapt gaussian", th3)
    #cv2.imshow("Canny", edges)
    #cv2.imshow("CannyDilated", dilate)
    #cv2.imshow("Adapt mean erosion", th2erosion)
    #cv2.imshow("Adapt gaussian erosion", th3erosion)
    #cv2.imshow("erosion", erosion)

    cv2.waitKey(1)



if __name__ == '__main__':
    # Adaptive gaussian or mean ( gaussian is a bit better )
    # gaussian a difficoltà sul molto scuro

    #img = cv2.imread(BASE_PATH + "Datasets/drAIver/line_detection/street.jpg")
    #img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000009.png")
    #img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000014.png")
    #img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000024.png")  # bad ( bad with -40)  <= very big problem
    img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000044.png") # problem to find correct two lines
    #img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000047.png") # more or less ( otzu very good here )
    #img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000071.png")  # more or less ( otzu very bad here )
    #img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000123.png")
    # linee trateggiate
    #img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000081.png")
    #img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000087.png")

    img = cv2.resize(img, (WIDTH, HEIGHT))

    # TODO fix linee tratteggiate

    detect(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

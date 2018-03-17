import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

HEIGHT = 480
WIDTH = 640

BASE_PATH = "/mnt/B01EEC811EEC41C8/"


def compute_theta(img,x1, y1, x2, y2):
    if x1-x2 == 0:
        x1 = x1 + 1 #TODO search better method
    return np.arctan((y1 - y2) / (x1 - x2))

def compute_m(x1, y1, x2, y2):

    return (y1 - y2) / (x1 - x2)

def compute_rho(img,x1, y1, x2, y2):
    m = compute_m(x1, y1, x2, y2)
    return abs(y1 + m*x1)/math.sqrt(pow(m, 2)+1)


def detect(img, negate = False):

    gray = np.zeros((HEIGHT, WIDTH, 1), dtype=np.uint8)
    cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, gray, 1)
    if negate:
        gray = abs(255 - gray)

    #value , thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU) # otzu works very bad with light conditions

    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -35) # maybe use a bit little biass
    #th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -15)

    #kernel = np.ones((3, 3), np.uint8)
    #th2erosion = cv2.erode(th2, kernel, iterations=1)
    #th3erosion = cv2.erode(th3, kernel, iterations=1)

    #edges = cv2.Canny(thr, 100, 200)







    #erosion = cv2.erode(thr, kernel, iterations=1)
    #dilate = cv2.dilate(edges, kernel, iterations=1)


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

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=3, lineType=cv2.LINE_8)

        if abs(theta) > 0.5: #TODO fix theta
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=3, lineType=cv2.LINE_8)

            filtered_lines.append((rho,theta))

    print("============ Filtered lines =============")

    rhos = []
    thetas = []
    for rho, theta in filtered_lines:
        thetas.append(theta)
        rhos.append(rho)

    plt.scatter(rhos, thetas)
    plt.show()

    line_points = [list(t) for t in zip(rhos, thetas)]

    kmeans = KMeans(n_clusters=2, random_state=0).fit(line_points)
    # >> > kmeans.labels_
    # array([0, 0, 0, 1, 1, 1], dtype=int32)
    # >> > kmeans.predict([[0, 0], [4, 4]])
    # array([0, 1], dtype=int32)
    # >> > kmeans.cluster_centers_
    # array([[1., 2.],
    #        [4., 2.]])

    for p,l in zip(line_points, kmeans.labels_):
        if l == 1 :
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
        #pt1 = (int(round(y0 + 1000 * (a))), int(round(x0 + 1000 * (-b))))
        pt2 = (int(round(x0 - 1000 * (-b))), int(round(y0 - 1000 * (a))))
        #pt2 = (int(round(y0 - 1000 * (a))), int(round(x0 - 1000 * (-b))))

        cv2.line(img, pt1, pt2, (128, 55, 23), thickness=3, lineType=cv2.LINE_8)

    plt.show()

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

    img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000009.png")
    # img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000014.png")
    # img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000024.png")  # bad ( bad with -40)  <= very big problem
    #img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000044.png")
    # img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000047.png") # more or less ( otzu very good here )
    # img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000071.png")  # more or less ( otzu very bad here )
    # img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000123.png")
    # linee trateggiate
    # img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000081.png")
    # img = cv2.imread(BASE_PATH + "Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000087.png")

    img = cv2.resize(img, (WIDTH, HEIGHT))

    detect(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

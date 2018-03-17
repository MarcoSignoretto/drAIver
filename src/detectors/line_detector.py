import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

HEIGHT = 480
WIDTH = 640

BASE_PATH = "/mnt/B01EEC811EEC41C8/"


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


    lines = cv2.HoughLinesP(th2, 1, np.pi/180, 130, minLineLength=30, maxLineGap=5)
    filtered_lines = []
    for line in lines:
        l = line[0]
        print(l)
        cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), thickness=3, lineType=cv2.LINE_8)





        # filter are for line detection
        height, width = img.shape[:2]
        if(l[1] > height/2 or l[3] > height/2):
            cv2.line(img, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), thickness=3, lineType=cv2.LINE_8)

            # filter for angle (search vertical)
            theta = compute_theta(l[0], l[1], l[2], l[3])
            print(theta)
            if (abs(theta) > 0.5): #TODO fix theta
                cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), thickness=3, lineType=cv2.LINE_8)

                filtered_lines.append(l)

    print("============ Filtered lines =============")

    rhos = []
    thetas = []
    for f_l in filtered_lines:
        thetas.append(compute_theta(f_l[0], f_l[1], f_l[2], f_l[3]))
        rhos.append(compute_rho(f_l[0], f_l[1], f_l[2], f_l[3]))
        print(f_l)

    plt.scatter(rhos, thetas)
    plt.show()

    line_points = [list(t) for t in zip(rhos, thetas)]

    kmeans = KMeans(n_clusters=10, random_state=0).fit(line_points)
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

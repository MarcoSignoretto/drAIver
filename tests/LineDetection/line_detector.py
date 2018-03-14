#!/envs/drAIver/bin/python

import cv2
import numpy as np

HEIGHT = 480
WIDTH = 640

def compute_theta(x1,y1,x2,y2):
    if x1-x2 == 0:
        x1 = x1 + 1 #TODO search better method
    return np.arctan((y1 - y2) / (x1 - x2))


def main():
    # Adaptive gaussian or mean ( gaussian is a bit better )
    # gaussian a difficoltà sul molto scuro

    #img = cv2.imread("/Users/marco/Documents/Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000009.png")
    img = cv2.imread("/Users/marco/Documents/Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000014.png")
    #img = cv2.imread("/Users/marco/Documents/Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000024.png")  # bad ( bad with -40)  <= very big problem
    #img = cv2.imread("/Users/marco/Documents/Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000044.png")
    #img = cv2.imread("/Users/marco/Documents/Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000047.png") # more or less ( otzu very good here )
    #img = cv2.imread("/Users/marco/Documents/Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000071.png")  # more or less ( otzu very bad here )
    #img = cv2.imread("/Users/marco/Documents/Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000123.png")
    # linee trateggiate
    #img = cv2.imread("/Users/marco/Documents/Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000081.png")
    #img = cv2.imread("/Users/marco/Documents/Datasets/drAIver/KITTY/data_object_image_2/training/image_2/000087.png")


    img = cv2.resize(img, (WIDTH, HEIGHT))

    gray = np.zeros((HEIGHT, WIDTH, 1), dtype=np.uint8)
    cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, gray, 1)

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
            theta = compute_theta(l[0], l[1], l[2], l[3]) #np.arctan((l[1] - l[3]) / (l[0] - l[2]))
            print(theta)
            if (abs(theta) > 0.5): #TODO fix theta
                cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), thickness=3, lineType=cv2.LINE_8)

                filtered_lines.append(l)

    print("============ Filtered lines =============")
    for f_l in filtered_lines:
        print(f_l)



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

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

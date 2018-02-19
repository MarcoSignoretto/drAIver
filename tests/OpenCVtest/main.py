import cv2


def main():
    image = cv2.imread("clouds.jpg")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Over the Clouds", image)
    cv2.imshow("Over the Clouds - gray", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




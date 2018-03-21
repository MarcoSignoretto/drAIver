#!/envs/drAIver/bin/python

import numpy as np
import cv2
import glob
import time
import draiver.camera.properties as cp

CAMERA = 1

CHESSBOARD_ROW_CORNERS = 6
CHESSBOARD_COL_CORNERS = 9

class BirdEye:
    pass

def find_anchors(corners):
    print(corners)






if __name__ == '__main__':
    # Open Camera
    vc = cv2.VideoCapture()
    vc.open(CAMERA)
    time.sleep(1)  # without this camera setup failed
    print(vc.set(cv2.CAP_PROP_FRAME_WIDTH, cp.FRAME_WIDTH))

    print(vc.set(cv2.CAP_PROP_FRAME_HEIGHT, cp.FRAME_HEIGHT))
    print(vc.set(cv2.CAP_PROP_FPS, cp.FPS))
    time.sleep(1)  # without this camera setup failed


    while True:

        if vc.isOpened():

            ret_left, img = vc.read()
            if ret_left:
                gray = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, gray, 1)
                value , thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU) # otzu works very bad with light conditions

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_ROW_CORNERS, CHESSBOARD_COL_CORNERS), None)
                if ret :
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(img, (CHESSBOARD_ROW_CORNERS, CHESSBOARD_COL_CORNERS), corners2, ret)
                #dst = cv2.cornerHarris(thr, 6, 11, 0.04)

                    points = np.empty([4, 2], dtype=np.float32)
                    destination_points = np.float32([
                        [cp.FRAME_WIDTH/CHESSBOARD_ROW_CORNERS, cp.FRAME_HEIGHT/CHESSBOARD_COL_CORNERS],
                        [cp.FRAME_WIDTH - (cp.FRAME_WIDTH/CHESSBOARD_ROW_CORNERS), cp.FRAME_HEIGHT/CHESSBOARD_COL_CORNERS],
                        [cp.FRAME_WIDTH/CHESSBOARD_ROW_CORNERS, cp.FRAME_HEIGHT - (cp.FRAME_HEIGHT/CHESSBOARD_COL_CORNERS)],
                        [cp.FRAME_WIDTH - (cp.FRAME_WIDTH/CHESSBOARD_ROW_CORNERS), cp.FRAME_HEIGHT - (cp.FRAME_HEIGHT/CHESSBOARD_COL_CORNERS)]]
                    )


                    #cv2.circle(img, (int(np.round(x_new)), K), 5, (134, 234, 100), thickness=2)
                    p = 0
                    for i in [0,5,48,53]:
                        c = corners2[i].flatten()
                        points[p] = c
                        p = p +1
                        x = c[0]
                        y = c[1]
                        cv2.circle(img, (x, y) ,3,(255,0,0), thickness=3)

                        M = cv2.getPerspectiveTransform(points, destination_points)
                        dst = cv2.warpPerspective(img, M, (640, 480))
                    # cv2.circle(img,corners2[5],3,(255,0,0), thickness=3)
                    # cv2.circle(img,corners2[53],3,(255,0,0), thickness=3)
                    find_anchors(corners2)

                # result is dilated for marking the corners, not important
                #dst = cv2.dilate(dst, None)



                cv2.imshow("Frame", img)
                cv2.moveWindow("Frame",100,100)

                cv2.imshow("TH", dst)
                cv2.moveWindow("TH", 800, 100)
                cv2.waitKey(1)


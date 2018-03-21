#!/envs/drAIver/bin/python

import numpy as np
import cv2
import glob
import time

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

ITERATIONS = 200
CAMERA = 1

CHESSBOARD_ROW_CORNERS = 6
CHESSBOARD_COL_CORNERS = 9


def detect_distortions():

    # Open Camera
    vc = cv2.VideoCapture()
    vc.open(CAMERA)
    time.sleep(1)  # without this camera setup failed
    print(vc.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH))

    print(vc.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT))
    print(vc.set(cv2.CAP_PROP_FPS, FPS))
    time.sleep(1)  # without this camera setup failed

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHESSBOARD_ROW_CORNERS * CHESSBOARD_COL_CORNERS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COL_CORNERS, 0:CHESSBOARD_ROW_CORNERS].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # images = glob.glob('*.jpg')

    it = 0
    while it < ITERATIONS:

        if vc.isOpened():

            ret_left, img = vc.read()
            if ret_left:

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Find the chess board corners
                print(CHESSBOARD_ROW_CORNERS,CHESSBOARD_COL_CORNERS)
                ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_ROW_CORNERS, CHESSBOARD_COL_CORNERS), None)

                # If found, add object points, image points (after refining them)
                if ret == True:
                    objpoints.append(objp)

                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)

                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (CHESSBOARD_ROW_CORNERS, CHESSBOARD_COL_CORNERS), corners2, ret)
                    print("Snap!!")
                    it = it + 1

                cv2.imshow('img', img)
                cv2.waitKey(1)

    cv2.destroyAllWindows()
    vc.release()
    return objpoints, imgpoints


def calibrate(objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (FRAME_WIDTH, FRAME_HEIGHT), None, None)
    return ret, mtx, dist, rvecs, tvecs


def undistort(img, ret, mtx, dist, rvecs, tvecs):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    #x, y, w, h = roi
    #dst = dst[y:y + h, x:x + w]
    #cv2.imwrite('calibresult.png', dst)

    # # undistort
    # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y + h, x:x + w]
    return dst


if __name__ == '__main__':
    objpoints, imgpoints = detect_distortions()
    ret, mtx, dist, rvecs, tvecs = calibrate(objpoints, imgpoints)

    vc = cv2.VideoCapture()
    vc.open(CAMERA)
    time.sleep(1)  # without this camera setup failed
    print(vc.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH))
    print(vc.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT))
    print(vc.set(cv2.CAP_PROP_FPS, FPS))
    time.sleep(1)  # without this camera setup failed

    if vc.isOpened():
        ret_left, img = vc.read()
        if ret_left:
            dst = undistort(img, ret, mtx, dist, rvecs, tvecs)

            cv2.imshow("Original", img)
            cv2.moveWindow("Original", 100, 100)
            cv2.imshow("Undistorted", dst)
            cv2.moveWindow("Undistorted", 800, 100)

            cv2.waitKey(0)






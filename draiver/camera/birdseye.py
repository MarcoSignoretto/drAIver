#!/envs/drAIver/bin/python

import numpy as np
import cv2
import time
import draiver.camera.properties as cp
import os

BIRDVIEW_HEIGHT_OFFSET = 200

def detect_camera_perspective_and_save(camera_index, perspective_file_path, external_camera):
    # Open Camera
    vc = cv2.VideoCapture()
    vc.open(camera_index)
    time.sleep(1)  # without this camera setup failed
    print(vc.set(cv2.CAP_PROP_FRAME_WIDTH, cp.FRAME_WIDTH))
    print(vc.set(cv2.CAP_PROP_FRAME_HEIGHT, cp.FRAME_HEIGHT))
    print(vc.set(cv2.CAP_PROP_FPS, cp.FPS))
    time.sleep(1)  # without this camera setup failed

    width = cp.FRAME_WIDTH
    height = cp.FRAME_HEIGHT

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    loop = True
    while loop:

        if vc.isOpened():

            ret_left, img = vc.read()
            if ret_left:
                gray = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, gray, 1)

                ret, corners = cv2.findChessboardCorners(gray, (cp.CHESSBOARD_ROW_CORNERS, cp.CHESSBOARD_COL_CORNERS), None)
                if ret:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(img, (cp.CHESSBOARD_ROW_CORNERS, cp.CHESSBOARD_COL_CORNERS), corners2, ret)
                    # dst = cv2.cornerHarris(thr, 6, 11, 0.04)

                    points = np.empty([4, 2], dtype=np.float32)
                    height = height + BIRDVIEW_HEIGHT_OFFSET
                    if external_camera:
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
                    else:
                        #  Robot destination points
                        destination_points = np.float32([
                            [
                                width / 3,
                                0,
                            ], [
                                width - (width / 3),
                                0
                            ], [
                                width / 3,
                                height# height - (height / 3)
                            ], [
                                width - (width / 3),
                                height# height - (height / 3)
                            ]
                        ])

                    p = 0
                    for i in [0, 5, 48, 53]:
                        c = corners2[i].flatten()
                        points[p] = c
                        p = p + 1
                        x = c[0]
                        y = c[1]
                        cv2.circle(img, (x, y), 3, (255, 0, 0), thickness=3)

                    M = cv2.getPerspectiveTransform(points, destination_points)
                    dst = cv2.warpPerspective(img, M, (width, height))

                    cv2.imshow("Frame", img)
                    cv2.moveWindow("Frame", 100, 100)

                    cv2.imshow("BirdView", dst)
                    cv2.moveWindow("BirdView", 800, 100)

                    cv2.waitKey(100)

                    command = input("Save camera perspective transformation? [Y,n,c]: ")
                    if command in ['y', '', 'Y']:

                        print("Saving Camera Perspective transform...")
                        np.save(perspective_file_path, M)
                        print("Saved")
                        loop = False

                    elif command in ['n', 'N']:
                        loop = False
                    elif command in ['c', 'C']:
                        loop = True
                    else:
                        print("Invalid option!")

                else:

                    cv2.imshow("Frame", img)
                    cv2.moveWindow("Frame", 100, 100)

                    cv2.waitKey(1)


class BirdsEye:

    def __init__(self, M = None, perspective_file_path=cp.DEFAULT_BIRDSEYE_CONFIG_PATH, width=cp.FRAME_WIDTH, height=cp.FRAME_HEIGHT, negate=False):
        self.width = width
        self.height = height
        if negate:
            self.border_value = 0
        else:
            self.border_value = 255
        if M is not None:
            self.M = M
        else:
            if os.path.isfile(perspective_file_path):
                # camera perspective transformation
                self.M = np.load(perspective_file_path)
            else:
                print("WARNING!!!! => Perspective camera information not ready.")

    def apply(self, img):
        return cv2.warpPerspective(img, self.M, (self.width, self.height + BIRDVIEW_HEIGHT_OFFSET), borderMode=cv2.BORDER_REPLICATE, borderValue=(self.border_value, self.border_value, self.border_value))






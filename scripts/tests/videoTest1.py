#!/envs/drAIver/bin/python

import cv2
import numpy as np
from draiver.camera.birdseye import BirdsEye
import draiver.camera.properties as cp
import draiver.detectors.line_detector_v3 as ld

DEBUG = False
PLOT = False

# BASE_PATH = "/mnt/B01EEC811EEC41C8/" # Ubuntu Config
BASE_PATH = "/Users/marco/Documents/"

VIDEO_PATH = "Datasets/drAIver/KITTY/2011_09_26/2011_09_26_drive_0027_sync/image_03/data/"

if __name__ == '__main__':
    # Adaptive gaussian or mean ( gaussian is a bit better )
    # gaussian a difficoltà sul molto scuro

    video_frames = 187

    for i in range(0, video_frames):
        file_name = format(i, '010d')
        path = VIDEO_PATH+file_name+'.png'
        print(path)

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
        img = cv2.resize(img, (cp.FRAME_WIDTH, cp.FRAME_HEIGHT))
        #img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.medianBlur(img, 3)  # remove noise from HS channels TODO choose

        # Work only on bird view
        if DEBUG:
            cv2.imshow("Frame", img)
        img = birdeye.apply(img)

        # ======================== DETECTION ===========================

        left, right = ld.detect(img)

        # ======================== PLOT ===========================

        if left is not None:
            for i in range(0, img.shape[0]-1):
                y_fit = left[0]*(i**2) + left[1]*i + left[2]
                cv2.circle(img, (int(y_fit), i), 1, (0, 0, 255), thickness=1)

        if right is not None:
            for i in range(0, img.shape[0] - 1):
                y_fit = right[0] * (i ** 2) + right[1] * i + right[2]
                cv2.circle(img, (int(y_fit), i), 1, (0, 0, 255), thickness=1)

        cv2.imshow("Frame d", img)
        cv2.moveWindow("Frame d", 100, 100)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
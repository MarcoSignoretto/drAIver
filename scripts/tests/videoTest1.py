#!/envs/drAIver/bin/python

import cv2
import numpy as np
from draiver.camera.birdseye import BirdsEye
import draiver.camera.properties as cp
import draiver.detectors.line_detector_v3 as ld

DEBUG = False
PLOT = False

BASE_PATH = "/mnt/B01EEC811EEC41C8/" # Ubuntu Config
#  BASE_PATH = "/Users/marco/Documents/"

VIDEO_PATH_1 = "Datasets/drAIver/KITTY/2011_09_26/2011_09_26_drive_0027_sync/image_03/data/"
# TODO bad !! difficult for dotted lines and light changes
VIDEO_PATH_2 = "Datasets/drAIver/KITTY/2011_09_26/2011_09_26_drive_0028_sync/image_02/data/"
# TODO 03 extreme light condition fails, with precise warping fixed!!
VIDEO_PATH_3 = "Datasets/drAIver/KITTY/2011_09_29/2011_09_29_drive_0004_sync/image_03/data/"
# TODO very difficult on dotted lines
VIDEO_PATH_4 = "Datasets/drAIver/KITTY/2011_09_26/2011_09_26_drive_0029_sync/image_03/data/"
# TODO need some thresholding adjustment + curved line is difficult if base start is hist start
VIDEO_PATH_5 = "Datasets/drAIver/KITTY/2011_10_03/2011_10_03_drive_0042_sync/image_03/data/"
# TODO not so bad


VIDEO_FRAMES_1 = 187
VIDEO_FRAMES_2 = 429
VIDEO_FRAMES_3 = 338
VIDEO_FRAMES_4 = 429
VIDEO_FRAMES_5 = 1169

if __name__ == '__main__':
    # Adaptive gaussian or mean ( gaussian is a bit better )
    # gaussian a difficoltà sul molto scuro
    video_path = VIDEO_PATH_1
    video_frames = VIDEO_FRAMES_1

    for i in range(0, video_frames):
        file_name = format(i, '010d')
        path = video_path+file_name+'.png'
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
                height + 200
            ], [
                width - (width / cp.CHESSBOARD_ROW_CORNERS),
                height + 200
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
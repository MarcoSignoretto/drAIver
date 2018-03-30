#!/envs/drAIver/bin/python
import cv2
import numpy as np
import draiver.detectors.line_detector_v3 as ld
from draiver.camera.birdseye import BirdsEye
import draiver.camera.properties as cp
import sys, getopt


def detect_lines(bird):
    # Line Detection
    left, right = ld.detect(bird, negate=True, robot=True)

    # ======================== PLOT ===========================

    if left is not None:
        for i in range(0, bird.shape[0] - 1):
            y_fit = left[0] * (i ** 2) + left[1] * i + left[2]
            cv2.circle(bird, (int(y_fit), i), 1, (0, 0, 255), thickness=1)

    if right is not None:
        for i in range(0, bird.shape[0] - 1):
            y_fit = right[0] * (i ** 2) + right[1] * i + right[2]
            cv2.circle(bird, (int(y_fit), i), 1, (0, 0, 255), thickness=1)


def main(camera_index):
    key = ''
    vc = cv2.VideoCapture(camera_index)
    print(vc.set(cv2.CAP_PROP_FRAME_WIDTH, cp.FRAME_WIDTH))
    print(vc.set(cv2.CAP_PROP_FRAME_HEIGHT, cp.FRAME_HEIGHT))
    print(vc.set(cv2.CAP_PROP_FPS, cp.FPS))

    width = cp.FRAME_WIDTH
    height = cp.FRAME_HEIGHT

    points = np.float32([
        [
            237,
            292
        ], [
            440,
            292
        ], [
            170,
            478
        ], [
            480,
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

    birdview = BirdsEye(M=M, negate=True)

    # birdview = BirdsEye(perspective_file_path="../../config/camera_perspective.npy", negate=True)

    while key != ord('q'):
        _,frame = vc.read()
        frame = cv2.medianBlur(frame, 3)

        bird = birdview.apply(frame)

        detect_lines(bird)

        cv2.imshow("Frame", frame)
        cv2.moveWindow("Frame", 10, 10)
        cv2.imshow("Bird", bird)
        cv2.moveWindow("Bird", 10+cp.FRAME_WIDTH, 10)

        key = cv2.waitKey(1) & 0xFF

    cv2.destroyAllWindows()

if __name__ == '__main__':
    camera_index = 0
    num_frames = 100
    display = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hc:f:d', ['help', 'camera=', 'num_frames=', 'display'])
    except getopt.GetoptError:
        print('camera_stream.py -c <camera_index> -f <num_frames> -d <y/n>')
        sys.exit(2)

    for o, a in opts:
        if o in ("-c", "--camera"):
            camera_index = int(a)
        elif o in ("-h", "--help"):
            sys.exit()
        else:
            assert False, "unhandled option"

    main(camera_index)

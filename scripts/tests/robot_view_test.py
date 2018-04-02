#!/envs/drAIver/bin/python
import cv2
import numpy as np
import draiver.detectors.line_detector_v3 as ld
from draiver.camera.birdseye import BirdsEye
import draiver.camera.properties as cp
import sys, getopt
import draiver.motion.steering as st
from draiver.util.queue import SkipQueue
from threading import Thread

queue = SkipQueue(1)

def capture_task(queue):

    vc = cv2.VideoCapture(camera_index)
    print(vc.set(cv2.CAP_PROP_FRAME_WIDTH, cp.FRAME_WIDTH))
    print(vc.set(cv2.CAP_PROP_FRAME_HEIGHT, cp.FRAME_HEIGHT))
    print(vc.set(cv2.CAP_PROP_FPS, cp.FPS))

    while True:
        _, frame = vc.read()
        queue.put(frame)


def image_task(queue):
    key = ''
    width = cp.FRAME_WIDTH
    height = cp.FRAME_HEIGHT

    birdview = BirdsEye(perspective_file_path="../../config/camera_perspective.npy", negate=True)

    while key != ord('q'):
        frame = queue.get()

        frame = cv2.medianBlur(frame, 3)

        bird = birdview.apply(frame)

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

        # ======================== CALCULATE INTERCEPT ===================

        intercept = height - 100

        left_int, right_int = st.find_intersection_points(left, right, intercept)

        # Plot

        if left_int is not None:
            cv2.circle(bird, (int(left_int), intercept), 1, (255, 0, 255), thickness=3)
        if right_int is not None:
            cv2.circle(bird, (int(right_int), intercept), 1, (255, 0, 255), thickness=3)

        # ======================== CAR POSITION ===================
        car_position = int(bird.shape[1] / 2)
        steering_range = 500
        mid = None
        if right_int is not None and left_int is not None:
            steering_range = right_int - left_int
            mid = left_int + steering_range / 2
        elif left_int is not None:
            mid = left_int + steering_range / 2
        elif right_int is not None:
            mid = right_int - steering_range / 2

        # plot
        cv2.circle(bird, (int(car_position), intercept), 1, (255, 0, 0), thickness=8)
        if mid is not None:
            cv2.circle(bird, (int(mid), intercept), 1, (13, 128, 255), thickness=5)

        cv2.imshow("Frame", frame)
        cv2.moveWindow("Frame", 10, 10)
        cv2.imshow("Bird", bird)
        cv2.moveWindow("Bird", 10 + cp.FRAME_WIDTH, 10)

        key = cv2.waitKey(1) & 0xFF

    cv2.destroyAllWindows()


def main(camera_index):

    capture_thread = Thread(target=capture_task, args=[queue])
    capture_thread.start()

    image_task(queue)


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

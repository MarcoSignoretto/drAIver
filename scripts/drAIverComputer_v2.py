#!/envs/drAIver/bin/python

import socket
import cv2
import numpy as np
from draiver.motion.motorcontroller import MotorController
import draiver.detectors.line_detector_v3 as ld
import draiver.camera.properties as cp
from draiver.camera.birdseye import BirdsEye

OUTPUT_PORT = 10001
INPUT_PORT = 10000

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

SPEED = 50

LINE_DETECTOR_NEGATE = True


def recvall(sock, count):

    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def image_task():

    motor_controller = MotorController()
    motor_controller.start()

    print("Image Thread Started")
    # socket init
    server_address = (socket.gethostbyname("drAIver.local"), INPUT_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server_address)



    key = ''
    while key != ord('q'):
        length_left = recvall(sock, 16)
        stringData_left = recvall(sock, int(length_left))
        data_left = np.fromstring(stringData_left, dtype='uint8')
        decimg_left = cv2.imdecode(data_left, 1)

        # TODO remove this section and use calibration file

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

        # ========== ELABORATE FRAME better if image sent to different thread =========
        frame = cv2.medianBlur(decimg_left, 3)

        bird = birdview.apply(frame)

        left, right = ld.detect(bird, negate=True, robot=True)
        car_position = int(bird.shape[1]/2)

        motor_controller.get_queue().put((left, right, car_position))

        #  ==================== End eleboration ================================

        cv2.imshow('CLIENT_LEFT', decimg_left)
        key = cv2.waitKey(1) & 0xFF

    cv2.destroyAllWindows()
    sock.close()


if __name__ == '__main__':

    image_task()










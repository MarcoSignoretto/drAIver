#!/envs/drAIver/bin/python

import socket
import cv2
import numpy as np
from draiver.motion.motorcontroller import MotorController
import draiver.detectors.line_detector as ld

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

        # ========== ELABORATE FRAME better if image sent to different thread =========
        left, right, car_position = ld.detect(decimg_left, negate=LINE_DETECTOR_NEGATE)

        motor_controller.get_queue().put((left, right, car_position))

        #  ==================== End eleboration ================================

        cv2.imshow('CLIENT_LEFT', decimg_left)
        key = cv2.waitKey(1) & 0xFF

    cv2.destroyAllWindows()
    sock.close()


if __name__ == '__main__':

    image_task()










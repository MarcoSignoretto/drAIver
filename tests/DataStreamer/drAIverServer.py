# This script must be executed on the drAIver robot
# Driver is the server because it has the access point configured

import socket
import cv2
import numpy as np
import time
from threading import Thread


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

OUTPUT_PORT = 10000
INPUT_PORT = 10001


def image_task():
    # socket init
    server_address = (socket.gethostbyname("drAIver.local"), OUTPUT_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(server_address)
    sock.listen(True)
    conn, addr = sock.accept()

    # camera init
    vc = cv2.VideoCapture()
    vc.open(0)
    time.sleep(1)  # without this camera setup failed
    print(vc.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH))
    print(vc.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT))
    print(vc.set(cv2.CAP_PROP_FPS, FPS))
    time.sleep(1)  # without this camera setup failed

    while True:

        if vc.isOpened():

            ret_left, frame_left = vc.read()
            if ret_left:
                # encodind frma in JPEG format to obtain better transmission speed
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                # =========== LEFT IMAGE ============
                result_left, imgencode_left = cv2.imencode('.jpg', frame_left, encode_param)

                # transform image in np.array
                data_left = np.array(imgencode_left)
                stringData_left = data_left.tostring()

                # send chunk
                conn.send(str(len(stringData_left)).ljust(16).encode())
                conn.send(stringData_left)

    cv2.destroyAllWindows()
    conn.close()


def motion_task():
    # socket init
    server_address = (socket.gethostbyname("drAIver.local"), INPUT_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(server_address)
    sock.listen(True)
    conn, addr = sock.accept()

    # TODO complete


if __name__ == '__main__':

    image_thread = Thread(target=image_task)
    image_thread.start()




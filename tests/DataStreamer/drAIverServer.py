# This script must be executed on the drAIver robot
# Driver is the server because it has the access point configured

import socket
import cv2
import numpy as np


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30


if __name__ == '__main__':
    # socket init
    server_address = (socket.gethostbyname("drAIver.local"), 10000)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(server_address)
    sock.listen(True)
    conn, addr = sock.accept()

    # camera init
    vc = cv2.VideoCapture()
    vc.open(0)
    print(vc.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH))
    print(vc.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT))
    print(vc.set(cv2.CAP_PROP_FPS, FPS))

    while True:

        ret_left, frame_left = vc.read()

        # This code should be removed
        #cv2.imshow('CAMERA LEFT', frame_left)
        #cv2.waitKey(1)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        # =========== LEFT IMAGE ============
        result_left, imgencode_left = cv2.imencode('.jpg', frame_left, encode_param)

        # With png we have lags
        # encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
        # result, imgencode = cv2.imencode('.png', frame, encode_param)
        data_left = np.array(imgencode_left)
        stringData_left = data_left.tostring()

        conn.send(str(len(stringData_left)).ljust(16).encode())
        conn.send(stringData_left)

    cv2.destroyAllWindows()
    conn.close()



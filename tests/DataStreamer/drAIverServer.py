# This script must be executed on the drAIver robot
# Driver is the server because it has the access point configured

import socket
import cv2
import numpy as np
import time
from threading import Thread
from motorprotocol import MotorProtocol
import brickpi3


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

OUTPUT_PORT = 10000
INPUT_PORT = 10001

COMMUNICATION_END = 0x01000000

def recvall(sock, count):

    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def image_task():
    print("Image Thread Started")
    # socket init
    server_address = (socket.gethostbyname("drAIver.local"), OUTPUT_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(server_address)
    sock.listen(True)
    print("Image task waiting...")
    conn, addr = sock.accept()
    print("Image task connected")

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
    print("Motion Thread Started")
    # socket init
    server_address = (socket.gethostbyname("drAIver.local"), INPUT_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(server_address)
    sock.listen(True)
    print("Motion task waiting ...")
    conn, addr = sock.accept()
    print("Motion task connected")

    BP = brickpi3.BrickPi3()

    mp = MotorProtocol()

    packet = int.from_bytes(recvall(conn, MotorProtocol.COMMUNICATION_PACKET_SIZE), byteorder='big') & MotorProtocol.COMMUNICATION_MASK
    while packet != COMMUNICATION_END:
        packet = int.from_bytes(recvall(conn, MotorProtocol.COMMUNICATION_PACKET_SIZE), byteorder='big') & MotorProtocol.COMMUNICATION_MASK
        print(packet)

        left_packet, right_packet = mp.split(packet)
        _, left_speed = mp.decompose(left_packet)
        _, right_speed = mp.decompose(right_packet)

        # TODO fix negative number

        print("LEFT: "+str(left_speed))
        print("RIGHT: "+str(right_speed))

        BP.set_motor_power(BP.PORT_D, left_speed)
        BP.set_motor_power(BP.PORT_A, right_speed)

    BP.reset_all()

    # TODO complete


if __name__ == '__main__':

    image_thread = Thread(target=image_task)
    image_thread.start()

    motion_thread = Thread(target=motion_task)
    motion_thread.start()




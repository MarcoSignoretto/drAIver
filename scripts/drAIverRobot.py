#!/envs/drAIver/bin/python

import socket
import cv2
import numpy as np
import time
from threading import Thread
from draiver.communication.motorprotocol import MotorProtocol
import brickpi3
from draiver.util.queue import SkipQueue
from queue import Empty


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

OUTPUT_PORT = 10000
INPUT_PORT = 10001

COMMUNICATION_END = 0xFFFF


global_sending_queue = SkipQueue(1)
global_motion_queue = SkipQueue(1)


def recvall(sock, count):

    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def image_sending(sending_queue):
    # socket init
    server_address = (socket.gethostbyname("drAIver.local"), OUTPUT_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(1)
    sock.bind(server_address)
    sock.listen(True)
    print("Image task waiting...")
    conn, addr = sock.accept()
    print("Image task connected")
    conn.setblocking(1)
    while True:
        stringData_left = sending_queue.get()
        # send chunk
        conn.send(str(len(stringData_left)).ljust(16).encode())
        conn.send(stringData_left)


def image_task(sending_queue):
    sock = None
    conn = None
    vc = cv2.VideoCapture()
    try:
        print("Image Thread Started")

        # camera init
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
                    sending_queue.put(stringData_left)
    except:
        print("Exception Image task")
    finally:
        cv2.destroyAllWindows()
        if conn is not None:
            conn.close()
        if sock is not None:
            sock.close()
        if vc.isOpened():
            vc.release()


def collect_motion_data():

    sock = None
    conn = None
    try:
        print("Motion Thread Receiver Started")
        # socket init
        server_address = (socket.gethostbyname("drAIver.local"), INPUT_PORT)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(server_address)
        sock.listen(True)
        print("Motion task waiting ...")
        conn, addr = sock.accept()
        print("Motion task connected")

        mp = MotorProtocol()

        packet = int.from_bytes(recvall(conn, MotorProtocol.COMMUNICATION_PACKET_SIZE), byteorder='big') & MotorProtocol.COMMUNICATION_MASK
        while packet != COMMUNICATION_END:
            packet = int.from_bytes(recvall(conn, MotorProtocol.COMMUNICATION_PACKET_SIZE), byteorder='big') & MotorProtocol.COMMUNICATION_MASK
            print(packet)

            left_speed, right_speed = mp.split(packet)
            global_motion_queue.put((left_speed, right_speed))



    except:
        print("Exception motion task")
    finally:
        if conn is not None:
            conn.close()
        if sock is not None:
            sock.close()


    # TODO complete

def motion_task():
    BP = None
    try:
        print("Motion Thread Started")
        BP = brickpi3.BrickPi3()

        while True:
            left_speed = 0
            right_speed = 0
            try:
                left_speed, right_speed = global_motion_queue.get(timeout=0.2)
            except Empty:
                pass
            finally:
                print("LEFT: " + str(left_speed))
                print("RIGHT: " + str(right_speed))
                BP.set_motor_power(BP.PORT_D, left_speed)
                BP.set_motor_power(BP.PORT_A, right_speed)

    except:
        print("Exception motion task")
    finally:
        if BP is not None:
            BP.reset_all()


def main():

    motion_data_thread = Thread(target=collect_motion_data)
    motion_data_thread.start()

    motion_thread = Thread(target=motion_task)
    motion_thread.start()

    sending_thread = Thread(target=image_sending, args=[global_sending_queue])
    sending_thread.start()

    image_thread = Thread(target=image_task, args=[global_sending_queue])
    image_thread.start()


if __name__ == '__main__':
    main()

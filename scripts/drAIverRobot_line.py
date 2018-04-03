#!/envs/drAIver/bin/python

import socket
import cv2
import numpy as np
import time
from threading import Thread
from draiver.communication.motorprotocol import MotorProtocol
import brickpi3
from draiver.util.queue import SkipQueue
from draiver.camera.birdseye import BirdsEye
import draiver.detectors.line_detector_v3 as ld
import draiver.motion.steering as st


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
        frame_left = sending_queue.get()

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        # =========== LEFT IMAGE ============
        result_left, imgencode_left = cv2.imencode('.jpg', frame_left, encode_param)

        # transform image in np.array
        data_left = np.array(imgencode_left)
        stringData_left = data_left.tostring()

        # send chunk
        conn.send(str(len(stringData_left)).ljust(16).encode())
        conn.send(stringData_left)


def image_task(sending_queue, motion_queue):
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

                    # send chunk
                    motion_queue.put(frame_left)
                    sending_queue.put(frame_left)
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


def motion_task():
    BP = None
    sock = None
    conn = None
    try:
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

            left_speed, right_speed = mp.split(packet)

            print("LEFT: "+str(left_speed))
            print("RIGHT: "+str(right_speed))

            BP.set_motor_power(BP.PORT_D, left_speed)
            BP.set_motor_power(BP.PORT_A, right_speed)

    except:
        print("Exception motion task")
    finally:
        if BP is not None:
            BP.reset_all()
        if conn is not None:
            conn.close()
        if sock is not None:
            sock.close()


def local_motion_task(motion_queue):

    birdview = BirdsEye(negate=True)
    BP = brickpi3.BrickPi3()

    while True:
        decimg_left = motion_queue.get()
        frame = cv2.medianBlur(decimg_left, 3)

        bird = birdview.apply(frame)

        left, right = ld.detect(bird, negate=True, robot=True)
        car_position = int(bird.shape[1] / 2)

        left_speed, right_speed = st.calculate_steering(left, right, car_position)

        BP.set_motor_power(BP.PORT_D, left_speed)
        BP.set_motor_power(BP.PORT_A, right_speed)


def main():

    #motion_thread = Thread(target=motion_task)
    #motion_thread.start()

    motion_thread = Thread(target=local_motion_task, args=[global_motion_queue])
    motion_thread.start()

    sending_thread = Thread(target=image_sending, args=[global_sending_queue])
    sending_thread.start()

    image_thread = Thread(target=image_task, args=[global_sending_queue, global_motion_queue])
    image_thread.start()


if __name__ == '__main__':
    main()

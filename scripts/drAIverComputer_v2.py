#!/envs/drAIver/bin/python

import socket
import cv2
import numpy as np
from draiver.motion.motorcontroller import MotorController
from draiver.communication.motorprotocol import MotorProtocol
import draiver.detectors.line_detector_v3 as ld
from draiver.util.queue import SkipQueue
from threading import Thread
from draiver.camera.birdseye import BirdsEye
import draiver.motion.steering as st
import time

OUTPUT_PORT = 10001
INPUT_PORT = 10000

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

SPEED = 50

LINE_DETECTOR_NEGATE = True

global_image_queue = SkipQueue(1)
global_motion_queue = SkipQueue(1)


def recvall(sock, count):

    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def collect_image_data():
    print("Image Thread Started")
    # socket init
    server_address = (socket.gethostbyname("drAIver.local"), INPUT_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server_address)

    while True:
        length_left = recvall(sock, 16)
        stringData_left = recvall(sock, int(length_left))
        data_left = np.fromstring(stringData_left, dtype='uint8')
        decimg_left = cv2.imdecode(data_left, 1)

        global_image_queue.put(decimg_left)
        print("Image arrived!!")

    sock.close()


def image_task():

    birdview = BirdsEye(negate=True)

    while True:
        frame = global_image_queue.get()
        print("Received image")

        frame = cv2.medianBlur(frame, 3)

        bird = birdview.apply(frame)

        left, right = ld.detect(bird, negate=True, robot=True)

        left_speed, right_speed = st.calculate_steering(bird, left, right)

        global_motion_queue.put((left_speed, right_speed))

        # ======================== PLOT ===========================

        if left is not None:
            for i in range(0, bird.shape[0] - 1):
                y_fit = left[0] * (i ** 2) + left[1] * i + left[2]
                cv2.circle(bird, (int(y_fit), i), 1, (0, 0, 255), thickness=1)

        if right is not None:
            for i in range(0, bird.shape[0] - 1):
                y_fit = right[0] * (i ** 2) + right[1] * i + right[2]
                cv2.circle(bird, (int(y_fit), i), 1, (0, 0, 255), thickness=1)


        cv2.imshow('CLIENT_LEFT', frame)
        cv2.moveWindow('CLIENT_LEFT', 10, 10)

        cv2.imshow('BIRD', bird)
        cv2.moveWindow('BIRD', 660, 10)
        key = cv2.waitKey(1) & 0xFF

    cv2.destroyAllWindows()


def motion_task():
    print("Motion Thread Started")
    # socket init
    server_address = (socket.gethostbyname("drAIver.local"), OUTPUT_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server_address)
    sock.setblocking(1)

    mp = MotorProtocol()

    while True:

        left_speed, right_speed = global_motion_queue.get()

        left_packet = mp.pack(left_speed)
        right_packet = mp.pack(right_speed)

        packet = mp.merge(left_packet, right_packet)
        sock.send(packet.to_bytes(MotorProtocol.COMMUNICATION_PACKET_SIZE, byteorder='big'))
        time.sleep(0.1)

    sock.close()


if __name__ == '__main__':

    motion_thread = Thread(target=motion_task)
    motion_thread.start()

    image_thread = Thread(target=collect_image_data)
    image_thread.start()

    image_task()









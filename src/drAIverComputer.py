#!/envs/drAIver/bin/python

import socket
import cv2
import numpy as np
import keyboard
from threading import Thread
from communication.motorprotocol import MotorProtocol
import time

OUTPUT_PORT = 10001
INPUT_PORT = 10000

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

SPEED = 50

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
    server_address = (socket.gethostbyname("drAIver.local"), INPUT_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server_address)

    key = ''
    while key != ord('q'):
        length_left = recvall(sock, 16)
        stringData_left = recvall(sock, int(length_left))
        data_left = np.fromstring(stringData_left, dtype='uint8')
        decimg_left = cv2.imdecode(data_left, 1)

        cv2.imshow('CLIENT_LEFT', decimg_left)
        key = cv2.waitKey(1) & 0xFF

    cv2.destroyAllWindows()
    sock.close()


def motion_task():
    print("Motion Thread Started")
    # socket init
    server_address = (socket.gethostbyname("drAIver.local"), OUTPUT_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server_address)

    mp = MotorProtocol()

    running = True
    while running:

        left_packet = mp.pack(0)
        right_packet = mp.pack(0)

        if keyboard.is_pressed('q'):
            running = False
            print("Stop!")

        if keyboard.is_pressed('e'):
            print("Motor Left Forth")
            left_packet = mp.pack(SPEED)

        elif keyboard.is_pressed('d'):
            print("Motor Left Back")
            left_packet = mp.pack(-SPEED)

        if keyboard.is_pressed('p'):
            print("Motor Right Forth")
            right_packet = mp.pack(SPEED)

        elif keyboard.is_pressed('l'):
            print("Motor Right Back")
            right_packet = mp.pack(-SPEED)

        packet = mp.merge(left_packet, right_packet)
        sock.send(packet.to_bytes(MotorProtocol.COMMUNICATION_PACKET_SIZE, byteorder='big'))
        time.sleep(0.3)

    sock.close()


if __name__ == '__main__':

    motion_thread = Thread(target=motion_task)
    motion_thread.start()

    image_task()










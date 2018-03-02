import socket
import cv2
import numpy as np
import keyboard
from threading import Thread
from motorprotocol import MotorProtocol

OUTPUT_PORT = 10001
INPUT_PORT = 10000

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30


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

    while True:
        #char = getch.getche()
        #print(keyboard.is_pressed(ord('q')))


        left_packet = mp.pack(MotorProtocol.MOTOR_LEFT, 0)
        right_packet = mp.pack(MotorProtocol.MOTOR_RIGHT, 0)


        if keyboard.is_pressed('q'):
            print("Stop!")
            exit(0)

        if keyboard.is_pressed('e'):
            print("Motor Left Forth")
            left_packet = mp.pack(MotorProtocol.MOTOR_LEFT, 50)
            #time.sleep(button_delay)

        elif keyboard.is_pressed('d'):
            print("Motor Left Back")
            left_packet = mp.pack(MotorProtocol.MOTOR_LEFT, -50)
            #time.sleep(button_delay)

        if keyboard.is_pressed('p'):
            print("Motor Right Forth")
            right_packet = mp.pack(MotorProtocol.MOTOR_RIGHT, 50)
            #time.sleep(button_delay)

        elif keyboard.is_pressed('l'):
            print("Motor Right Back")
            right_packet = mp.pack(MotorProtocol.MOTOR_RIGHT, -50)
            #time.sleep(button_delay)

        packet = mp.merge(left_packet, right_packet)
        sock.send(packet.to_bytes(MotorProtocol.COMMUNICATION_PACKET_SIZE, byteorder='big'))


if __name__ == '__main__':

    image_thread = Thread(target=image_task)
    image_thread.start()
    #
    motion_thread = Thread(target=motion_task)
    motion_thread.start()
    # motion_task()

    # left_motor_thread = Thread(target=left_motor_task)
    # left_motor_thread.start()
    #
    # right_motor_thread = Thread(target=right_motor_task)
    # right_motor_thread.start()










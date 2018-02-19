#!/usr/bin/python
import socket
import cv2
import numpy

CAMERA_LEFT = 0
CAMERA_RIGHT = 1

# camera left init
camera_left = cv2.VideoCapture()
camera_left.set(4, 640)
camera_left.set(5, 480)
camera_left.open(CAMERA_LEFT)
# camera right init
camera_right = cv2.VideoCapture()
camera_right.set(4, 640)
camera_right.set(5, 480)
camera_right.open(CAMERA_RIGHT)

# socket init
server_address = ('10.42.0.1', 10000)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(server_address)
while True:
    ret_left, frame_left = camera_left.read()
    ret_right, frame_right = camera_right.read()

    # This code should be removed
    cv2.imshow('CAMERA LEFT', frame_left)
    cv2.imshow('CAMERA RIGHT', frame_right)
    cv2.waitKey(1)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    # =========== LEFT IMAGE ============
    result_left, imgencode_left = cv2.imencode('.jpg', frame_left, encode_param)

    # With png we have lags
    #encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
    #result, imgencode = cv2.imencode('.png', frame, encode_param)
    data_left = numpy.array(imgencode_left)
    stringData_left = data_left.tostring()

    sock.send(str(len(stringData_left)).ljust(16).encode())
    sock.send(stringData_left)
    # =========== RIGHT IMAGE =============
    result_right, imgencode_right = cv2.imencode('.jpg', frame_right, encode_param)

    # With png we have lags
    # encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
    # result, imgencode = cv2.imencode('.png', frame, encode_param)
    data_right = numpy.array(imgencode_right)
    stringData_right = data_right.tostring()

    sock.send(str(len(stringData_right)).ljust(16).encode())
    sock.send(stringData_right)

    #decimg = cv2.imdecode(data_left, 1)
    #cv2.imshow('ORIGINAL', frame)

    #cv2.imwrite('frame.png', frame)
    #cv2.imwrite('frame.jpg', decimg)


cv2.destroyAllWindows()
sock.close()
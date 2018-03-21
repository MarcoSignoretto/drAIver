#!/usr/bin/python
import socket
import cv2
import numpy

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

# camera init
camera_left = cv2.VideoCapture()
camera_left.set(4, 640)
camera_left.set(5, 480)
camera_left.open(0)

# socket init
server_address = ('10.42.0.1', 10000)
sock=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(server_address)
sock.listen(True)
conn, addr = sock.accept()
while True:
    length_left = recvall(conn,16)
    stringData_left = recvall(conn, int(length_left))
    data_left = numpy.fromstring(stringData_left, dtype='uint8')
    decimg_left=cv2.imdecode(data_left,1)

    length_right = recvall(conn, 16)
    stringData_right = recvall(conn, int(length_right))
    data_right = numpy.fromstring(stringData_right, dtype='uint8')
    decimg_right = cv2.imdecode(data_right, 1)



    cv2.imshow('SERVER_LEFT',decimg_left)
    cv2.imshow('SERVER_RIGHT',decimg_right)
    cv2.waitKey(1)
cv2.destroyAllWindows()
sock.close()
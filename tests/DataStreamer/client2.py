import cv2
import numpy as np
import socket
import sys
import pickle
import struct

# camera init
camera_left = cv2.VideoCapture()
camera_left.set(4, 640)
camera_left.set(5, 480)
camera_left.open(0)

# socket init
server_address = ('10.42.0.1', 10000)
clientsocket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(server_address)
while True:
    ret, frame = camera_left.read()
    # cv2.imshow("camera_left", frame)
    # cv2.waitKey(1)
    cv2.waitKey(300)
    data = pickle.dumps(frame)

    #_, buf = cv2.imencode('.jpg', frame)
    # clientsocket.sendall(struct.pack("I", len(buf))+buf)
    clientsocket.sendall(struct.pack("I", len(data))+data)

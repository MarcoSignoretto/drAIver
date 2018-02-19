import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new

import time

HOST='10.42.0.1'
PORT=10000

s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn, addr = s.accept()

data = b''
payload_size = struct.calcsize("I")
while True:
    # Read payload
    ts_payload = time.time()
    while len(data) < payload_size:
        data += conn.recv(4096)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("I", packed_msg_size)[0]
    te_payload = time.time()
    # Read data
    ts_data = time.time()
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    te_data = time.time()
    ###
    ts_frame = time.time()
    frame = pickle.loads(frame_data)
    # print(frame)
    cv2.imshow('frame', frame)
    te_frame = time.time()
    cv2.waitKey(1)

    print("payload: "+str(te_payload-ts_payload)+", data: "+str(te_data-ts_data)+", frame: "+str(te_frame-ts_frame))

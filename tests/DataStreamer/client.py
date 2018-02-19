
# socket_echo_client_dgram.py

import socket
import sys
import cv2
import pickle
import struct

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_address = ('10.42.0.1', 10000)
message = b'This is the message.  It will be repeated.'

camera_left = cv2.VideoCapture()

camera_left.set(4, 640)
camera_left.set(5, 480)


camera_left.open(0)

if camera_left.isOpened():
    _, frame_left = camera_left.read()
    cv2.imshow("camera_left", frame_left)

cv2.waitKey(0)

PACKET_SIZE = 4096

try:

    i = 0
    bytes_sent = 0

    flat_frame = frame_left.flatten()
    s = flat_frame.tostring()
    while bytes_sent < len(flat_frame):
        packet = s[i * PACKET_SIZE:(i + 1) * PACKET_SIZE]
        sent = sock.sendto(packet, server_address)
        i = i + 1
        bytes_sent = bytes_sent + PACKET_SIZE


    #sent = sock.sendto(flat_frame, server_address)  ### new code


    # Send data
    print('sending {!r}'.format(message))
    sent = sock.sendto(message, server_address)

    # Receive response
    print('waiting to receive')
    data, server = sock.recvfrom(PACKET_SIZE) # 4096
    print('received {!r}'.format(data))

finally:
    print('closing socket')
    sock.close()


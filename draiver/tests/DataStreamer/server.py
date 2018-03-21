# server.py

import socket
import sys
import numpy
import cv2

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the port
server_address = ('10.42.0.1', 10000)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)

PACKET_SIZE = 4096

s = b''
packet_number = 0

while True:

    print('\nwaiting to receive message')
    data, address = sock.recvfrom(PACKET_SIZE)
    s += data
    packet_number = packet_number + 1

    print(len(s))
    print('paqcket number' + str(packet_number))

    if len(s) == 913450:
        frame = numpy.fromstring(s, dtype=numpy.uint8)
        frame = frame.reshape(480, 640, 3)

        cv2.imshow('frame', frame)




        # print('received {} bytes from {}'.format(
        #     len(data), address))
        # print(data)
        #
        # if data:
        #     sent = sock.sendto(data, address)
        #     print('sent {} bytes back to {}'.format(
        #         sent, address))








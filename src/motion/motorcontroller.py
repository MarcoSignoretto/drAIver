#!/envs/drAIver/bin/python

from queue import Queue
from threading import Thread
from draiver.communication.motorprotocol import MotorProtocol
import socket

BASE_SPEED = 20
STEERING_SCALE = 10


class MotorController(Thread):

    def __init__(self):
        self.queue = Queue()
        self.mp = MotorProtocol()
        Thread.__init__(self)

        # init socket connection
        # socket init
        server_address = (socket.gethostbyname("drAIver.local"), 10001)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(server_address)


    def get_queue(self):
        return self.queue

    def run(self):
        while True:
            left, right, car_position = self.queue.get()
            left_speed, right_speed = self.calculate_steering(left, right, car_position)
            self.send_steering(left_speed, right_speed)

    def calculate_steering(self, left, right, car_position):
        range = right-left
        mid = left + range/2

        left_speed = BASE_SPEED
        right_speed = BASE_SPEED

        car_offset = abs(car_position - mid)
        delta = (STEERING_SCALE * car_offset) / (range/2)

        if car_position > mid: #  go to left
            right_speed = right_speed + delta
            left_speed = left_speed - delta
        else:
            right_speed = right_speed - delta
            left_speed = left_speed + delta

        return left_speed, right_speed

    def send_steering(self, left_speed, right_speed):

        left_packet = self.mp.pack(left_speed)
        right_packet = self.mp.pack(right_speed)

        packet = self.mp.merge(left_packet, right_packet)
        self.sock.send(packet.to_bytes(MotorProtocol.COMMUNICATION_PACKET_SIZE, byteorder='big'))


if __name__ == '__main__':
    pass






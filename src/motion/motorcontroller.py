#!/envs/drAIver/bin/python

from queue import Queue
from threading import Thread
from draiver.communication.motorprotocol import MotorProtocol

BASE_SPEED = 20
STEERING_SCALE = 10


class MotorController(Thread):

    def __init__(self):
        self.queue = Queue()
        self.mp = MotorProtocol()
        Thread.__init__(self)

    def get_queue(self):
        return self.queue

    def run(self):
        while True:
            left, right, car_position = self.queue.get()
            self.calculate_steering(left, right, car_position)
            print(left, right, car_position)

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

        print(left_speed, right_speed)


if __name__ == '__main__':
    pass






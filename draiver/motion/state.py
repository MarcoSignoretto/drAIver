#!/envs/drAIver/bin/python
from queue import Queue
from threading import Lock
import draiver.camera.properties as cp
import numpy as np


class DrivingState:

    def __init__(self, perspective_file_path=cp.DEFAULT_BIRDSEYE_CONFIG_PATH):
        self.perspective_transform = np.load(perspective_file_path)
        self.lock = Lock()
        self.last_car_detections = [] # TODO solid detection like car, pedestrians and so on
        self.last_sign_detections = [] # TODO solid detection like car, pedestrians and so on

        self.last_base_speed = 0

        self.last_left_line = None
        self.last_right_line = None

        self.last_steering_delta = None
        self.last_steering_car_position = None
        self.last_steering_mid = None
        self.actions = Queue()

    def get_base_speed(self):
        with self.lock:
            # TODO control on detections
            return self.last_base_speed

    def set_car_detections(self, detections):
        # TODO for each detection associate an action ( stop if detection under certain distance
        with self.lock:
            self.last_car_detections = detections
            for det in self.last_car_detections:
                # TODO is under certain threshold set action
                pass


    def get_car_detections(self):
        # TODO for each detection associate an action ( stop if detection under certain distance, hadle previus detection )
        with self.lock:
            return self.last_car_detections

    def set_sign_detections(self, detections):
        with self.lock:
            pass

    def get_sign_detections(self):
        with self.lock:
            return self.last_sign_detections

    def set_lines(self, left, right):
        with self.lock:
            self.last_left_line = left
            self.last_right_line = right

    def get_lines(self):
        return self.last_left_line, self.last_right_line

    def set_steering(self, delta, car_position, mid):
        with self.lock:
            self.last_steering_delta = delta
            self.last_steering_car_position = car_position
            self.last_steering_mid = mid

    def get_perspective_transform(self):
        return self.perspective_transform

    def compute_motion_informations(self):
        # TODO as result compute the final motor commands
        with self.lock:
            pass



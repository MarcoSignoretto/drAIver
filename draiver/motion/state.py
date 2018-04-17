#!/envs/drAIver/bin/python
from threading import Lock
import draiver.camera.properties as cp
import numpy as np

DEFAULT_OBJECT_COLLISION_DISTANCE = 300
DEFAULT_FRAME_WITHOUT_DETECTION = 10

ACTION_COLLISION_AVOIDANCE = 'collision_avoidance'


# Standard way priority based on distance
PRIORITY_HIGH = 1
PRIORITY_MEDIUM = 10
PRIORITY_LOW = 100


class DrivingState:

    def __init__(self, bird_size, frame_size, perspective_file_path=cp.DEFAULT_BIRDSEYE_CONFIG_PATH, object_collision_distance=DEFAULT_OBJECT_COLLISION_DISTANCE):
        self.bird_height, self.bird_width = bird_size
        self.frame_height, self.frame_width = frame_size
        self.object_collision_distance = object_collision_distance

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
        self.avoid_collision = False
        self.frame_without_detection = 0

    def get_object_collision_distance(self):
        return self.object_collision_distance

    def get_base_speed(self):
        with self.lock:
            # TODO control on detections
            return self.last_base_speed

    def set_car_detections(self, detections):
        # TODO for each detection associate an action ( stop if detection under certain distance
        with self.lock:
            self.last_car_detections = detections
            detection_origins = np.zeros((3, len(self.last_car_detections)), dtype=np.float32)
            for i in range(0, len(self.last_car_detections)):
                det = self.last_car_detections[i]
                detection_origins[:, i] = [
                    det['bottomright']['x'],
                    det['bottomright']['y'],
                    1
                ]

            if len(self.last_car_detections) > 0:
                detection_origins_bird = np.matmul(self.perspective_transform, detection_origins)
                detection_origins_bird = detection_origins_bird / detection_origins_bird[2]
                max_index = np.argmax(detection_origins_bird[1])  # closest object

                distance = self.bird_height - detection_origins_bird[1, max_index]
                self.avoid_collision = distance < self.object_collision_distance
            elif self.frame_without_detection > DEFAULT_FRAME_WITHOUT_DETECTION:
                self.frame_without_detection = 0
                self.avoid_collision = False

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
            if self.avoid_collision:
                print("Stop!!! ")
                return 0, 0
            else:
                pass
                return 12, 12
                # TODO setup here



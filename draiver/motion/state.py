#!/envs/drAIver/bin/python
from threading import Lock
import draiver.camera.properties as cp
import numpy as np
import draiver.motion.steering as st
import draiver.util.detectorutil as du

DEFAULT_OBJECT_COLLISION_DISTANCE = 300
DEFAULT_FRAME_WITHOUT_DETECTION = 5

COLLISION_AVOIDANCE_ACTION_SET = ['car']#'collision_avoidance'

ACTION_COLLISION_AVOIDANCE = 'collision_avoidance'
ACTION_STOP = 'stop'
ACTION_PEDESTRIAN_CROSSING = 'pedestrianCrossing'
ACTION_AVOID_STOP = 'avoid_stop'

BASE_SPEED = 20

# Standard way priority based on distance
PRIORITY_HIGH = 1
PRIORITY_MEDIUM = 10
PRIORITY_LOW = 100

COLLISION_AVOIDANCE_DURATION_FRAMES = 50
STOP_DURATION_FRAMES = 50
AVOID_STOP_DURATION_FRAMES = 100
PEDESTRIAN_CROSSING_DURATION_FRAMES = 100


class DrivingState:

    def __init__(self, bird_size, frame_size, perspective_file_path=cp.DEFAULT_BIRDSEYE_CONFIG_PATH, object_collision_distance=DEFAULT_OBJECT_COLLISION_DISTANCE):
        self.bird_height, self.bird_width = bird_size
        self.frame_height, self.frame_width = frame_size
        self.object_collision_distance = object_collision_distance

        self.perspective_transform = np.load(perspective_file_path)
        self.lock = Lock()
        self.last_car_detections = [] # TODO solid detection like car, pedestrians and so on
        self.last_sign_detections = [] # TODO traffic sign detection

        self.last_base_speed = BASE_SPEED

        self.last_left_line = None
        self.last_right_line = None

        self.last_steering_delta = None
        self.last_steering_car_position = None
        self.last_steering_mid = None
        self.avoid_collision = False
        self.frame_without_collision_detection = 0

        self.actions = {}  # put key of the action and duration

    def get_object_collision_distance(self):
        return self.object_collision_distance

    def get_base_speed(self):
        with self.lock:
            # TODO control on detections
            return self.last_base_speed

    # def set_car_detections(self, detections):
    #     # TODO for each detection associate an action ( stop if detection under certain distance
    #     with self.lock:
    #         self.last_car_detections = detections
    #         detection_origins = np.zeros((3, len(self.last_car_detections)), dtype=np.float32)
    #         detection_ends = np.zeros((3, len(self.last_car_detections)), dtype=np.float32)
    #         for i in range(0, len(self.last_car_detections)):
    #             det = self.last_car_detections[i]
    #             detection_origins[:, i] = [
    #                 det['bottomright']['x'],
    #                 det['bottomright']['y'],
    #                 1
    #             ]
    #             # TODO understand if this is good
    #             # detection_ends[:, i] = [
    #             #     det['topleft']['x'],
    #             #     det['bottomright']['y'],
    #             #     1
    #             # ]
    #
    #         if len(self.last_car_detections) > 0:
    #             detection_origins_bird = np.matmul(self.perspective_transform, detection_origins)
    #             detection_origins_bird = detection_origins_bird / detection_origins_bird[2]
    #
    #             # TODO understand if this is good
    #             # detection_ends_bird = np.matmul(self.perspective_transform, detection_ends)
    #             # detection_ends_bird = detection_ends_bird / detection_ends_bird[2]
    #
    #             max_index = np.argmax(detection_origins_bird[1])  # closest object
    #
    #             distance = self.bird_height - detection_origins_bird[1, max_index]
    #
    #             # TODO understand if this is good
    #             # if self.last_left_line is not None and self.last_right_line is not None:
    #             #     int_left = st.find_intersection_point(self.last_left_line, detection_origins_bird[1, max_index])
    #             #     int_right = st.find_intersection_point(self.last_right_line, detection_origins_bird[1, max_index])
    #             #
    #             #     if detection_origins_bird[0, max_index] in range(int_left, int_right) and detection_ends_bird[0, max_index] in range(int_left,int_right):
    #             #         self.avoid_collision = distance < self.object_collision_distance
    #             # else: # No line detected so any object is in valid range
    #             #     self.avoid_collision = distance < self.object_collision_distance
    #
    #             self.avoid_collision = distance < self.object_collision_distance
    #
    #
    #         elif self.frame_without_collision_detection > DEFAULT_FRAME_WITHOUT_DETECTION:
    #             self.frame_without_collision_detection = 0
    #             self.avoid_collision = False

    def set_car_detections(self, detections):
        self.last_car_detections = detections
        for det in detections:
            det_label = du.find_class_detection(det)
            if det_label in COLLISION_AVOIDANCE_ACTION_SET:  # TODO test better distance from sign
                self.actions[ACTION_COLLISION_AVOIDANCE] = COLLISION_AVOIDANCE_DURATION_FRAMES

    def get_car_detections(self):
        with self.lock:
            return self.last_car_detections

    def set_sign_detections(self, detections):
        with self.lock:
            self.last_sign_detections = detections  # only for rendering
            for det in detections:
                det_label = du.find_class_detection(det)
                if det_label == ACTION_STOP and ACTION_STOP not in self.actions.keys() and ACTION_AVOID_STOP not in self.actions.keys():  # TODO test better distance from sign
                    self.actions[ACTION_STOP] = STOP_DURATION_FRAMES
                elif det_label == ACTION_PEDESTRIAN_CROSSING:  # while sees sign and for a period slow down
                    self.actions[ACTION_PEDESTRIAN_CROSSING] = PEDESTRIAN_CROSSING_DURATION_FRAMES
                # TODO yield not work

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

    def decrease_action(self, action_key):
        if action_key is not None:
            if self.actions[action_key] <= 0:
                self.actions.pop(action_key)
            else:
                self.actions[action_key] = self.actions[action_key] - 1

    # def compute_motion_informations(self):
    #     with self.lock:
    #         if self.avoid_collision:
    #             print("Stop Car!!! ")
    #             return 0, 0
    #         else:  # check other actions
    #             self.frame_without_collision_detection = self.frame_without_collision_detection + 1
    #
    #             if ACTION_STOP in self.actions.keys():
    #                 self.decrease_action(ACTION_STOP)
    #                 print("Stop Sign!!! ")
    #                 return 0, 0
    #             elif ACTION_PEDESTRIAN_CROSSING in self.actions.keys():
    #                 print("Pedestrian crossing!!! ")
    #                 self.decrease_action(ACTION_PEDESTRIAN_CROSSING)
    #                 self.last_base_speed = self.last_base_speed / 2.0
    #
    #             left_speed, right_speed = st.calculate_motor_speed_for_steering(
    #                 self.last_steering_delta,
    #                 self.last_steering_car_position,
    #                 self.last_steering_mid,
    #                 self.last_base_speed
    #             )
    #             print("Speed: %s %s" % (str(round(left_speed)), str(round(right_speed))))
    #             return left_speed, right_speed

    def compute_motion_informations(self):
        with self.lock:
            if ACTION_COLLISION_AVOIDANCE in self.actions.keys():
                self.decrease_action(ACTION_COLLISION_AVOIDANCE)
                print("Collision avoidance!! ")
                return 0, 0
            if ACTION_AVOID_STOP in self.actions.keys():
                self.decrease_action(ACTION_AVOID_STOP)

            if ACTION_STOP in self.actions.keys():
                if self.actions[ACTION_STOP] <= 0:
                    self.actions[ACTION_AVOID_STOP] = AVOID_STOP_DURATION_FRAMES
                self.decrease_action(ACTION_STOP)
                print("Stop Sign!!! ")
                return 0, 0
            elif ACTION_PEDESTRIAN_CROSSING in self.actions.keys():
                print("Pedestrian crossing!!! ")
                if self.actions[ACTION_PEDESTRIAN_CROSSING] <= 0:
                    self.last_base_speed = BASE_SPEED
                else:
                    self.last_base_speed = BASE_SPEED / 2.0
                self.decrease_action(ACTION_PEDESTRIAN_CROSSING)


            left_speed, right_speed = st.calculate_motor_speed_for_steering(
                self.last_steering_delta,
                self.last_steering_car_position,
                self.last_steering_mid,
                self.last_base_speed
            )
            print("Speed: %s %s" % (str(round(left_speed)), str(round(right_speed))))
            return left_speed, right_speed




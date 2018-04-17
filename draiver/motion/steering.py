#!/envs/drAIver/bin/python

import cv2

DEBUG = False
BASE_SPEED = 20
STEERING_SCALE = 30
MAX_STEERING = 30


def find_intersection_point(line, intercept):
    if line is not None:
        return int(line[0] * (intercept ** 2) + line[1] * intercept + line[2])
    else:
        return None


def find_intersection_points(line_left, line_right, intercept):
    left_int = find_intersection_point(line_left, intercept)
    right_int = find_intersection_point(line_right, intercept)
    return left_int, right_int


def calculate_steering_delta(bird_frame, left, right, horizon_line=100, steering_range=300):
    # ======================== CALCULATE INTERCEPT ===================
    delta = None
    intercept = bird_frame.shape[0] - horizon_line

    left_int, right_int = find_intersection_points(left, right, intercept)

    # Plot
    if DEBUG:
        if left_int is not None:
            cv2.circle(bird_frame, (int(left_int), intercept), 1, (255, 0, 255), thickness=3)
        if right_int is not None:
            cv2.circle(bird_frame, (int(right_int), intercept), 1, (255, 0, 255), thickness=3)

    # ======================== CAR POSITION ===================
    car_position = int(bird_frame.shape[1] / 2)
    mid = None
    if right_int is not None and left_int is not None:
        if right_int - left_int != 0:
            steering_range = right_int - left_int
        mid = left_int + steering_range / 2
    elif left_int is not None:
        mid = left_int + steering_range / 2
    elif right_int is not None:
        mid = right_int - steering_range / 2

    # plot
    if DEBUG:
        cv2.circle(bird_frame, (int(car_position), intercept), 1, (255, 0, 0), thickness=8)
        if mid is not None:
            cv2.circle(bird_frame, (int(mid), intercept), 1, (13, 128, 255), thickness=5)
        else:
            print("MID not found!!")

    if mid is not None:
        car_offset = min(abs(car_position - mid), STEERING_SCALE)
        delta = (STEERING_SCALE * car_offset) / (steering_range / 2.0)

    return delta, car_position, mid


def calculate_motor_speed_for_steering(delta, car_position, mid, base_speed):

    if delta is not None:

        left_speed = base_speed - (delta / 2.0)
        right_speed = base_speed - (delta / 2.0)

        if car_position > mid:  # go to left
            right_speed = right_speed + delta
            left_speed = left_speed - delta
        else:
            right_speed = right_speed - delta
            left_speed = left_speed + delta
    else:
        left_speed = 0
        right_speed = 0

    return left_speed, right_speed


def calculate_steering(bird_frame, left, right, horizon_line=100, steering_range=300):

    # # ======================== CALCULATE INTERCEPT ===================
    #
    # intercept = bird_frame.shape[0] - horizon_line
    #
    # left_int, right_int = find_intersection_points(left, right, intercept)
    #
    # # Plot
    # if DEBUG:
    #     if left_int is not None:
    #         cv2.circle(bird_frame, (int(left_int), intercept), 1, (255, 0, 255), thickness=3)
    #     if right_int is not None:
    #         cv2.circle(bird_frame, (int(right_int), intercept), 1, (255, 0, 255), thickness=3)
    #
    # # ======================== CAR POSITION ===================
    # car_position = int(bird_frame.shape[1] / 2)
    # mid = None
    # if right_int is not None and left_int is not None:
    #     if right_int - left_int != 0:
    #         steering_range = right_int - left_int
    #     mid = left_int + steering_range / 2
    # elif left_int is not None:
    #     mid = left_int + steering_range / 2
    # elif right_int is not None:
    #     mid = right_int - steering_range / 2
    #
    #
    #
    # # plot
    # if DEBUG:
    #     cv2.circle(bird_frame, (int(car_position), intercept), 1, (255, 0, 0), thickness=8)
    #     if mid is not None:
    #         cv2.circle(bird_frame, (int(mid), intercept), 1, (13, 128, 255), thickness=5)
    #     else:
    #         print("MID not found!!")
    #
    # # ======================= CALCULATE STEERING ==============
    #
    # left_speed = BASE_SPEED
    # right_speed = BASE_SPEED
    #
    # if mid is not None:
    #     car_offset = min(abs(car_position - mid), STEERING_SCALE)
    #     delta = (STEERING_SCALE * car_offset) / (steering_range / 2.0)
    #
    #     # exp_delta = (delta ** 2)/((STEERING_SCALE**2)/(MAX_STEERING))
    #     # if delta >= 0:
    #     #     delta = exp_delta
    #     # else:
    #     #     delta = -exp_delta
    delta, car_position, mid = calculate_steering_delta(bird_frame, left, right, horizon_line, steering_range)
    left_speed, right_speed = calculate_motor_speed_for_steering(delta, car_position, mid, BASE_SPEED)

    return left_speed, right_speed



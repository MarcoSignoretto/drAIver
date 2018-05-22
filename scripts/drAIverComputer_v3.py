#!/envs/drAIver/bin/python

import socket
import cv2
import numpy as np
from draiver.motion.motorcontroller import MotorController
from draiver.communication.motorprotocol import MotorProtocol
import draiver.detectors.line_detector_v3 as ld
from draiver.util.queue import SkipQueue
from threading import Thread
from draiver.camera.birdseye import BirdsEye
import draiver.motion.steering as st
from draiver.motion.state import DrivingState
import time
from draiver.detectors.objectdetector import SignDetector
from draiver.detectors.objectdetector import CarDetector
import draiver.camera.properties as cp
import sys, getopt
import draiver.util.drawing as dr

OUTPUT_PORT = 10001
INPUT_PORT = 10000

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

LINE_DETECTOR_NEGATE = True

global_line_detection_queue = SkipQueue(1)
global_car_detection_queue = SkipQueue(1)
global_sign_detection_queue = SkipQueue(1)
global_motion_queue = SkipQueue(1)
global_rendering_queue = SkipQueue(1)

driving_state = DrivingState((cp.BIRD_HEIGHT, cp.BIRD_WIDTH), (cp.FRAME_HEIGHT, cp.FRAME_WIDTH))


def recvall(sock, count):

    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def collect_image_data():
    print("Image Thread Started")
    # socket init
    server_address = (socket.gethostbyname("drAIver.local"), INPUT_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server_address)

    while True:
        length_left = recvall(sock, 16)
        stringData_left = recvall(sock, int(length_left))
        data_left = np.fromstring(stringData_left, dtype='uint8')
        decimg_left = cv2.imdecode(data_left, 1)

        # put frame into input queue of interested thread
        global_line_detection_queue.put(decimg_left)
        global_car_detection_queue.put(decimg_left)
        global_sign_detection_queue.put(decimg_left)
        global_rendering_queue.put(decimg_left)
        print("Image arrived!!")

    sock.close()


def collect_image_local_camera():

    vc = cv2.VideoCapture(1)
    print(vc.set(cv2.CAP_PROP_FRAME_WIDTH, cp.FRAME_WIDTH))
    print(vc.set(cv2.CAP_PROP_FRAME_HEIGHT, cp.FRAME_HEIGHT))
    print(vc.set(cv2.CAP_PROP_FPS, cp.FPS))

    while True:
        _, frame = vc.read()

        # put frame into input queue of interested thread
        global_line_detection_queue.put(frame)
        global_car_detection_queue.put(frame)
        global_sign_detection_queue.put(frame)
        global_rendering_queue.put(frame)
        print("Image arrived!!")


def sign_detection_task():
    sign_detector = SignDetector()
    while True:
        frame = global_sign_detection_queue.get()
        detection_result = sign_detector.detect(frame)
        driving_state.set_sign_detections(detection_result)


def car_detection_task():
    car_detection = CarDetector()
    while True:
        frame = global_car_detection_queue.get()
        detection_result = car_detection.detect(frame)
        print(detection_result)
        driving_state.set_car_detections(detection_result)


def line_detection_task():

    birdview = BirdsEye(negate=True)

    while True:
        frame = global_line_detection_queue.get()
        print("Received image")

        frame = cv2.medianBlur(frame, 3)

        bird = birdview.apply(frame)

        left, right = ld.detect(bird, negate=True, robot=True, thin=True)
        delta, car_position, mid = st.calculate_steering_delta(bird, left, right)
        #left_speed, right_speed = st.calculate_steering(bird, left, right)

        driving_state.set_lines(left, right)
        driving_state.set_steering(delta, car_position, mid)

        # global_motion_queue.put((left_speed, right_speed))



def fake_motion_task():

    while True:
        left_speed, right_speed = driving_state.compute_motion_informations()



def motion_task():
    print("Motion Thread Started")
    # socket init
    server_address = (socket.gethostbyname("drAIver.local"), OUTPUT_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server_address)
    sock.setblocking(1)

    mp = MotorProtocol()

    while True:

        #left_speed, right_speed = global_motion_queue.get()
        left_speed, right_speed = driving_state.compute_motion_informations()

        left_packet = mp.pack(left_speed)
        right_packet = mp.pack(right_speed)

        packet = mp.merge(left_packet, right_packet)
        sock.send(packet.to_bytes(MotorProtocol.COMMUNICATION_PACKET_SIZE, byteorder='big'))
        time.sleep(0.1)  # Refresh rate

    sock.close()


def rendering_task():

    birdview = BirdsEye(negate=True)

    while True:
        frame = global_rendering_queue.get()

        bird = birdview.apply(frame)

        left, right = driving_state.get_lines()

        # ======================== PLOT LINES ===========================

        if left is not None:
            for i in range(0, bird.shape[0] - 1):
                y_fit = left[0] * (i ** 2) + left[1] * i + left[2]
                cv2.circle(bird, (int(y_fit), i), 1, (0, 0, 255), thickness=1)

        if right is not None:
            for i in range(0, bird.shape[0] - 1):
                y_fit = right[0] * (i ** 2) + right[1] * i + right[2]
                cv2.circle(bird, (int(y_fit), i), 1, (0, 0, 255), thickness=1)

        # ======================== PLOT BOUNDING BOX =====================

        # M = np.linalg.inv(driving_state.get_perspective_transform())
        M = driving_state.get_perspective_transform()
        # TODO use np.linalg.inv( for conversion from bird to original
        # == car
        car_results = driving_state.get_car_detections()
        for res in car_results:
            dr.draw_detection(frame, res)

            detection_origin = np.array([
                [res['bottomright']['x']],
                [res['bottomright']['y']],
                [1]
            ])
            detection_origin_bird = np.matmul(M, detection_origin)
            detection_origin_bird = detection_origin_bird / detection_origin_bird[2]
            print("Detection origin bird:" + str(detection_origin_bird))
            #
            cv2.circle(bird, (detection_origin_bird[0], detection_origin_bird[1]), 2, (123, 0, 255), thickness=4)


        # == sign
        sign_results = driving_state.get_sign_detections()
        for res in sign_results:
            dr.draw_detection(frame, res)

        # ================= COLLISION LINE PLOT

        collision_y = bird.shape[0] - driving_state.get_object_collision_distance()
        cv2.line(bird, (0, collision_y), (bird.shape[1], collision_y), (255, 0, 0), thickness=2, lineType=cv2.LINE_8)





        cv2.imshow('CLIENT_LEFT', frame)
        cv2.moveWindow('CLIENT_LEFT', 10, 10)

        cv2.imshow('BIRD', bird)
        cv2.moveWindow('BIRD', 660, 10)
        key = cv2.waitKey(1) & 0xFF

    cv2.destroyAllWindows()


def main(local = False):
    if not local:
        # =====Robot ======
        motion_thread = Thread(target=motion_task)
        motion_thread.start()
        image_thread = Thread(target=collect_image_data)
        image_thread.start()
    else:
        # ===== Local camera =====
        motion_thread = Thread(target=fake_motion_task)
        motion_thread.start()
        image_thread = Thread(target=collect_image_local_camera)
        image_thread.start()

    line_detection_thread = Thread(target=line_detection_task)
    line_detection_thread.start()

    car_detection_thread = Thread(target=car_detection_task)
    car_detection_thread.start()

    sign_detection_thread = Thread(target=sign_detection_task)
    sign_detection_thread.start()

    rendering_task()


if __name__ == '__main__':
    local = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'l', ['local'])
    except getopt.GetoptError:
        print('drAiverComputer_v3.py -l')
        sys.exit(2)

    for o, a in opts:
        if o in ("-l", "--local"):
            local = True
        else:
            assert False, "unhandled option"

    main(local)














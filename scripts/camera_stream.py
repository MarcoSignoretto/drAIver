#!/envs/drAIver/bin/python

import cv2
import sys, getopt

FRAME_WIDTH = 640#320
FRAME_HEIGHT = 480#240

def main(camera_index):
    key = ''
    vc = cv2.VideoCapture(camera_index)
    print(vc.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH))
    print(vc.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT))
    print(vc.set(cv2.CAP_PROP_FPS, 30))

    while key != ord('q'):
        _,frame = vc.read()
        cv2.imshow("FRame", frame)
        key = cv2.waitKey(1) & 0xFF

    cv2.destroyAllWindows()

def usage():
    pass


if __name__ == '__main__':
    camera_index = 0
    num_frames = 100
    display = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hc:f:d', ['help', 'camera=', 'num_frames=', 'display'])
    except getopt.GetoptError:
        print('camera_stream.py -c <camera_index> -f <num_frames> -d <y/n>')
        sys.exit(2)

    for o, a in opts:
        if o in("-c","--camera"):
            camera_index = int(a)
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        else:
            assert False, "unhandled option"

    main(camera_index)




#!/envs/drAIver/bin/python

import draiver.camera.birdseye as be
import sys, getopt

if __name__ == '__main__':

    camera_index = 0
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hc:', ['help', 'camera='])
    except getopt.GetoptError:
        print('initCameraPerspective.py -c <camera_index> ')
        sys.exit(2)

    for o, a in opts:
        if o in ("-c", "--camera"):
            camera_index = int(a)
        elif o in ("-h", "--help"):
            print('initCameraPerspective.py -c <camera_index> ')
            sys.exit()
        else:
            assert False, "unhandled option"

    be.detect_camera_perspective_and_save(camera_index)

#!/envs/drAIver/bin/python

import draiver.camera.birdseye as be
import draiver.camera.properties as cp
import sys, getopt

if __name__ == '__main__':

    camera_index = 0
    external_cam = False
    config_path = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hc:p:e', ['help', 'camera=', 'config_path=', 'externalcamera'])
    except getopt.GetoptError:
        print('initCameraPerspective.py -c <camera_index> ')
        sys.exit(2)

    for o, a in opts:
        if o in ("-c", "--camera"):
            camera_index = int(a)
        elif o in ("-h", "--help"):
            print('initCameraPerspective.py -c <camera_index> -p <perspective_file_path> -e ')
            sys.exit()
        elif o in ("-p", "--config_path"):
            config_path = str(a)
        elif o in ("-e", "--externalcamera"):
            external_cam = True
        else:
            assert False, "unhandled option"

    if config_path is None:
        if not external_cam:
            config_path = cp.DEFAULT_BIRDSEYE_CONFIG_PATH
        else:
            config_path = cp.EXTERNAL_BIRDSEYE_CONFIG_PATH

    be.detect_camera_perspective_and_save(camera_index, config_path, external_cam)

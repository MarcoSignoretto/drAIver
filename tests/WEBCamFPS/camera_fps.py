import cv2
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import sys, getopt
import imutils

FRAME_WIDTH = 640#320
FRAME_HEIGHT = 480#240

def threadind():
    key = ''

    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()

    # loop over some frames...this time using the threaded stream
    while key != ord('q'):
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        # frame = imutils.resize(frame, width=400)

        # check to see if the frame should be displayed to our screen

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

def queueVersion():
    pass

def basic(camera_index,num_frames,display):
    key = ''
    vc = cv2.VideoCapture(camera_index)
    print(vc.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH))
    print(vc.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT))
    print(vc.set(cv2.CAP_PROP_FPS, 60))

    fps = FPS().start()
    steps = 0
    while key != ord('q') and steps < num_frames:
        _,frame = vc.read()
        if display:
            cv2.imshow("FRame",frame)
        key = cv2.waitKey(1) & 0xFF
        fps.update()
        steps = steps + 1

    fps.stop()
    cv2.destroyAllWindows()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


def usage():
    pass


if __name__ == '__main__':
    camera_index = 0
    num_frames = 100
    display = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hc:f:d', ['help', 'camera=', 'num_frames=', 'display'])
    except getopt.GetoptError:
        print('camera_fps.py -c <camera_index> -f <num_frames> -d <y/n>')
        sys.exit(2)

    for o, a in opts:
        if o in("-c","--camera"):
            camera_index = int(a)
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-f", "--num_frames"):
            num_frames = int(a)
        elif o in ("-d", "--display"):
            display = True
        else:
            assert False, "unhandled option"

    basic(camera_index, num_frames, display)




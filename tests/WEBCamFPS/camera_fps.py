import cv2
from imutils.video import WebcamVideoStream
from imutils.video import FPS
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

def basic():
    key = ''
    vc = cv2.VideoCapture(0)
    print(vc.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH))
    print(vc.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT))
    print(vc.set(cv2.CAP_PROP_FPS, 60))

    _, frame = vc.read()
    cv2.imshow("FRame", frame)
    fps = FPS().start()
    while key != ord('q'):
        _,frame = vc.read()
        cv2.imshow("FRame",frame)
        key = cv2.waitKey(1) & 0xFF
        fps.update()

    fps.stop()
    cv2.destroyAllWindows()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))



if __name__ == '__main__':
    basic()




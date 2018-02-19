import cv2
import numpy


CAMERA_LEFT = 1
CAMERA_RIGHT = 2#3

FRAME_WIDTH = 640#320
FRAME_HEIGHT = 480#240

def main():
    print(cv2.__version__)

    camera_left = cv2.VideoCapture(CAMERA_LEFT)
    camera_right = cv2.VideoCapture(CAMERA_RIGHT)

    #320 240
    #640 480

    print(camera_left.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH))
    print(camera_left.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT))
    print(camera_right.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH))
    print(camera_right.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT))

    #camera_left.open(CAMERA_LEFT)
    #camera_right.open(CAMERA_RIGHT)

    picture_taken = 0

    #while(picture_taken < 20):
    while True:

        if camera_left.isOpened():
            _, frame_left = camera_left.read()
            cv2.imshow("camera_left", frame_left)
            #print(frame_left.shape)+
        if camera_right.isOpened():
            _, frame_right = camera_right.read()
            cv2.imshow("camera_right", frame_right)

        cv2.waitKey(1)
        #
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        # # =========== LEFT IMAGE ============
        # result_left, imgencode_left = cv2.imencode('.jpg', frame_left, encode_param)
        # data_left = numpy.array(imgencode_left)
        # decimg_left = cv2.imdecode(data_left, 1)
        #
        # # =========== RIGHT IMAGE =============
        # result_right, imgencode_right = cv2.imencode('.jpg', frame_right, encode_param)
        # data_right = numpy.array(imgencode_right)
        # decimg_right = cv2.imdecode(data_right, 1)
        #
        # #cv2.imwrite('frame_left.jpg', decimg_left)
        # #cv2.imwrite('frame_right.jpg', decimg_right)
        # #cv2.waitKey(0)

        # cv2.imwrite('left_' + str(picture_taken) + '.png', frame_left)
        # cv2.imwrite('right_' + str(picture_taken) + '.png', frame_right)
        # picture_taken = picture_taken + 1
        #
        # cv2.waitKey(0)



    print("END")

    # Setup frame with fixed resolution in order to be fast and work in the same way for each resolution
    # vc.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    # vc.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    #
    # int frame_number = 1;
    #
    # while (!end) {
    # cv::
    #     Mat
    # camera_frame;
    #
    # vc >> camera_frame; // put
    # frame
    # image
    # Mat, this
    # operation is blocking
    # until
    # a
    # frame is provided
    # if (camera_frame.empty())
    # { // if camera
    # frame
    # are
    # infinite
    # but in video
    # no
    # end = true;
    # } else {
    #
    #     mcv:: marker::apply_AR(img_0p, img_1p, img_0m_th, img_1m_th, camera_frame, debug_info);
    # cv::imshow("original", camera_frame);
    #
    # cv::waitKey(1); // delay
    # between
    # frame
    # read( if to
    # long
    # we
    # loose
    # frames )
    # ++frame_number;
    # }
    #
    # }
    # } else {
    # std::cout << "Error impossible open the video" << std::endl;


if __name__ == '__main__':
    main()

from darkflow.net.build import TFNet
import cv2

options = {"model": "/Users/marco/Documents/GitProjects/UNIVE/darkflow/cfg/tiny-yolo-voc.cfg", "load": "/Users/marco/Documents/GitProjects/UNIVE/darkflow/bin/tiny-yolo-voc.weights", "threshold": 0.1}


FRAME_WIDTH = 640#320
FRAME_HEIGHT = 480#240

if __name__ == '__main__':

    camera = cv2.VideoCapture(0)
    print(camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH))
    print(camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT))

    tfnet = TFNet(options)

    #imgcv = cv2.imread("/Users/marco/Documents/Datasets/drAIver/object_detector/test_images/image1.jpg")

    #print(results)

    while True:
        if camera.isOpened():
            _, imgcv = camera.read()

            results = tfnet.return_predict(imgcv)

            for result in results:
                if result['confidence'] > 0.5:
                    cv2.rectangle(imgcv,
                                  (result['topleft']['x'], result['topleft']['y']),
                                  (result['bottomright']['x'], result['bottomright']['y']),
                                  (255, 0, 0)
                                  )

                cv2.imshow("Image", imgcv)
                cv2.waitKey(1)








import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf

from queue import Queue
from threading import Thread
from utils.app_utils import FPS, WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

import sys

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


sys.path.append("..")

#DATASET_PATH = '/Users/marco/Documents/Datasets/drAIver/object_detector/'
DATASET_PATH = 'S:\\Datasets\\drAIver\\object_detector\\'
BACK_SLASH = '\\'

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = DATASET_PATH + MODEL_NAME + BACK_SLASH +'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    #with tf.device('/device:CPU:0'): // TODO try on giulia PC
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    return dict(rect_points=boxes[0], class_names=classes[0].astype(np.uint8), class_scores=scores[0], num_detections=int(num_detections[0]))

    # # Visualization of the results of a detection.
    # rect_points, class_names, class_colors = draw_boxes_and_labels(
    #     boxes=np.squeeze(boxes),
    #     classes=np.squeeze(classes).astype(np.int32),
    #     scores=np.squeeze(scores),
    #     category_index=category_index,
    #     min_score_thresh=.5
    # )
    # return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=detection_graph, config=config)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    args = parser.parse_args()

    input_q = Queue(1)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()

    while True:
        frame = video_capture.read()
        input_q.put(frame)

        t = time.time()

        if output_q.empty():
            pass  # fill up queue
        else:



            font = cv2.FONT_HERSHEY_SIMPLEX
            data = output_q.get()

            #== My part of code

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                data['rect_points'],
                data['class_names'],
                data['class_scores'],
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            #==

            # rec_points = data['rect_points']
            # class_names = data['class_names']
            # class_colors = data['class_colors']
            # for point, name, color in zip(rec_points, class_names, class_colors):
            #     cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
            #                   (int(point['xmax'] * args.width), int(point['ymax'] * args.height)), color, 3)
            #     cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
            #                   (int(point['xmin'] * args.width) + len(name[0]) * 6,
            #                    int(point['ymin'] * args.height) - 10), color, -1, cv2.LINE_AA)
            #     cv2.putText(frame, name[0], (int(point['xmin'] * args.width), int(point['ymin'] * args.height)), font,
            #                 0.3, (0, 0, 0), 1)
            cv2.imshow('Video', frame)

        fps.update()

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    video_capture.stop()
    cv2.destroyAllWindows()
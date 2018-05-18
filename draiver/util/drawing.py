import cv2
import draiver.util.detectorutil as du

font = cv2.FONT_HERSHEY_SIMPLEX


def draw_detection(img, detection):
    color = (0, 255, 0)
    l_point1 = (int(detection['topleft']['x']), int(detection['topleft']['y'] - 30))
    l_point2 = (int(detection['topleft']['x'] + len(detection['label']) * 10), int(detection['topleft']['y']))
    cv2.rectangle(img, l_point1, l_point2, color, thickness=-1, lineType=cv2.LINE_8)
    cv2.putText(img, detection['label'], (int(detection['topleft']['x']), int(detection['topleft']['y'] - 10)), font, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)
    pt1 = (int(detection['topleft']['x']), int(detection['topleft']['y']))
    pt2 = (int(detection['bottomright']['x']), int(detection['bottomright']['y']))
    cv2.rectangle(img, pt1, pt2, color, thickness=2, lineType=cv2.LINE_8)


def draw_gt(img, annotation):
    color = (255, 0, 0)
    box = du.find_box_gt(annotation)
    label = du.find_class_gt(annotation)
    l_point1 = (int(box[0]), int(box[1] - 30))
    l_point2 = (int(box[0] + len(label) * 10), int(box[1]))
    cv2.rectangle(img, l_point1, l_point2, color, thickness=-1, lineType=cv2.LINE_8)
    cv2.putText(img, label, (int(box[0]), int(box[1] - 10)), font,
                0.5, (0, 0, 0), 1,
                cv2.LINE_AA)

    pt1 = (int(box[0]), int(box[1]))
    pt2 = (int(box[2]), int(box[3]))
    cv2.rectangle(img, pt1, pt2, color, thickness=2, lineType=cv2.LINE_8)


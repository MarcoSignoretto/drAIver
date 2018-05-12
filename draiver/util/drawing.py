import cv2

font = cv2.FONT_HERSHEY_SIMPLEX


def draw_detection(img, detection):
    l_point1 = (int(detection['topleft']['x']), int(detection['topleft']['y'] - 30))
    l_point2 = (int(detection['topleft']['x'] + len(detection['label']) * 10), int(detection['topleft']['y']))
    cv2.rectangle(img, l_point1, l_point2, (0, 255, 0), thickness=-1, lineType=cv2.LINE_8)
    cv2.putText(img, detection['label'], (int(detection['topleft']['x']), int(detection['topleft']['y'] - 10)), font, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)
    pt1 = (int(detection['topleft']['x']), int(detection['topleft']['y']))
    pt2 = (int(detection['bottomright']['x']), int(detection['bottomright']['y']))
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=2, lineType=cv2.LINE_8)


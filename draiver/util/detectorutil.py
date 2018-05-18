import numpy as np


def find_box_detection(detection):
    return np.array([
        float(detection['topleft']['x']),
        float(detection['topleft']['y']),
        float(detection['bottomright']['x']),
        float(detection['bottomright']['y'])
    ])


def find_class_detection(detection):
    return str(detection['label'])


def find_box_gt(xml_obj):
    xmin = float(xml_obj.find('bndbox').find('xmin').text)
    xmax = float(xml_obj.find('bndbox').find('xmax').text)
    ymin = float(xml_obj.find('bndbox').find('ymin').text)
    ymax = float(xml_obj.find('bndbox').find('ymax').text)
    return np.array([
        xmin,
        ymin,
        xmax,
        ymax
    ])


def find_class_gt(xml_obj):
    return str(xml_obj.find('name').text)

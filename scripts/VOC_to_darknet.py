#!/envs/drAIver/bin/python

from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

DATASET = "kitty_train_test"
# DATASET = "lisa"

TYPE = "test"

IMAGES = "images_" + TYPE
ANNOTATIONS = "annotations_" + TYPE

BASE_PATH = "/Users/marco/Documents/"

DATASET_PATH = BASE_PATH + "GitProjects/UNIVE/darkflow/training/"+DATASET+"/"

ANNOTATIONS_INPUT = DATASET_PATH + ANNOTATIONS + "/"
IMAGE_INPUT = DATASET_PATH + IMAGES+ "/"
ANNOTATIONS_OUTPUT = DATASET_PATH + "labels_%s/" %TYPE
CLASSES_FILE_PATH = DATASET_PATH+"labels.txt"

# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

if not os.path.exists(ANNOTATIONS_OUTPUT):
    os.makedirs(ANNOTATIONS_OUTPUT)

classes = []

classes_files = open(CLASSES_FILE_PATH).read().strip().split()
for c in classes_files:
    classes.append(c)


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convert_annotation(filename_no_ext):

    in_file = open('%s%s.xml'%(ANNOTATIONS_INPUT, filename_no_ext))
    out_file = open('%s%s.txt'%(ANNOTATIONS_OUTPUT, filename_no_ext), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# wd = getcwd()


files = [f for f in listdir(ANNOTATIONS_INPUT) if isfile(join(ANNOTATIONS_INPUT, f))]
files = [f for f in files if ".xml" in f]

list_file = open('%s%s_sources.txt' %(DATASET_PATH, IMAGES), 'w')
for filename in files:
    filename_no_ext = filename.replace('.xml', '')
    list_file.write('%s%s.png\n' % (IMAGE_INPUT,filename_no_ext))
    convert_annotation(filename_no_ext)
    print("filename: %s" %filename_no_ext)

list_file.close()

print("=========== FINISCH =============")


# for year, image_set in sets:
#     if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
#         os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
#     image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
#     list_file = open('%s_%s.txt'%(year, image_set), 'w')
#     for image_id in image_ids:
#         list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
#         convert_annotation(year, image_id)
#     list_file.close()
#
# os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
# os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")

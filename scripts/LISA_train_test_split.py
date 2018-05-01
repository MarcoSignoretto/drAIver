#!/envs/drAIver/bin/python
from os import listdir
from os.path import isfile, join
from random import randint
import os


BASE_PATH = "/Users/marco/Documents/"

LISA_PATH = BASE_PATH+"GitProjects/UNIVE/darkflow/training/lisa_train_test/"
IMAGES = LISA_PATH+"images/"
IMAGES_TRAIN = LISA_PATH+"images_train/"
IMAGES_TEST = LISA_PATH+"images_test/"

ANNOTATIONS = LISA_PATH+"annotations/"
ANNOTATIONS_TRAIN = LISA_PATH+"annotations_train/"
ANNOTATIONS_TEST = LISA_PATH+"annotations_test/"

TEST_PART = 0.3 # 30% test part

def main():
    images = [f for f in listdir(IMAGES) if isfile(join(IMAGES, f))]
    images = [f for f in images if ".png" in f]

    if not os.path.exists(IMAGES_TEST):
        os.makedirs(IMAGES_TEST)

    if not os.path.exists(ANNOTATIONS_TEST):
        os.makedirs(ANNOTATIONS_TEST)

    test_items = int(len(images)*0.3)
    for i in range(0, test_items):
        images = [f for f in listdir(IMAGES) if isfile(join(IMAGES, f))]
        images = [f for f in images if ".png" in f]

        item_index = randint(0, len(images)-1)

        filename = images[item_index]
        os.rename(IMAGES+filename, IMAGES_TEST+filename)

        filenameXml = filename.replace('.png', '.xml')
        os.rename(ANNOTATIONS + filenameXml, ANNOTATIONS_TEST + filenameXml)

        print(filename, filenameXml)

    os.rename(IMAGES, IMAGES_TRAIN)
    os.rename(ANNOTATIONS, ANNOTATIONS_TRAIN)

    print("FINISH")


if __name__ == '__main__':
    main()

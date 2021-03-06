#!/envs/drAIver/bin/python
from os import listdir
from os.path import isfile, join
from random import randint
import os
import draiver.env as env

BASE_PATH = env.DATASETS_HOME

DATASET = "lisa_train_test"

DATASET_PATH = BASE_PATH + DATASET + "/"
IMAGES = DATASET_PATH + "images/"
IMAGES_TRAIN = DATASET_PATH + "images_train/"
IMAGES_TEST = DATASET_PATH + "images_test/"

ANNOTATIONS = DATASET_PATH + "annotations/"
ANNOTATIONS_TRAIN = DATASET_PATH + "annotations_train/"
ANNOTATIONS_TEST = DATASET_PATH + "annotations_test/"

TEST_PART = 0.3 # 30% test part


def main():
    images = [f for f in listdir(IMAGES) if isfile(join(IMAGES, f))]
    images = [f for f in images if ".png" in f]
    images = [f for f in images if '._' not in f]  # Avoid mac issues

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

        filename_xml = filename.replace('.png', '.xml')
        os.rename(ANNOTATIONS + filename_xml, ANNOTATIONS_TEST + filename_xml)

        print(filename, filename_xml)

    os.rename(IMAGES, IMAGES_TRAIN)
    os.rename(ANNOTATIONS, ANNOTATIONS_TRAIN)

    print("FINISH")


if __name__ == '__main__':
    main()

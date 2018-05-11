#!/envs/drAIver/bin/python
import csv
import xml.etree.cElementTree as ET

LISA_CSV = "Datasets/drAIver/LISA/annotations/vid_annotations.csv"
# BASE_PATH = "/mnt/B01EEC811EEC41C8/" # Ubuntu Config
BASE_PATH = "/Users/marco/Documents/"

ANNOTATIONS_OUTPUT_DIR = "/Users/marco/Documents/GitProjects/UNIVE/darkflow/training/lisa/annotations/"


def new_object(root, row):
    object = ET.SubElement(root, "object")

    ET.SubElement(object, "name").text = row[1]

    bndbox = ET.SubElement(object, "bndbox")

    ET.SubElement(bndbox, "xmin").text = row[2]
    ET.SubElement(bndbox, "ymin").text = row[3]
    ET.SubElement(bndbox, "xmax").text = row[4]
    ET.SubElement(bndbox, "ymax").text = row[5]

    tree = ET.ElementTree(root)
    return tree


def build_xml_tree(row):
    root = ET.Element("annotation")

    k = row[0].rfind("/") + 1
    filename = row[0][k:len(row[0])]
    ET.SubElement(root, "filename").text = filename

    size = ET.SubElement(root, "size")

    # Fixed for LISA dataset
    ET.SubElement(size, "width").text = "704"
    ET.SubElement(size, "height").text = "480"
    ET.SubElement(size, "depth").text = "3"

    return new_object(root, row)


def main():
    prev_file = ''

    head = True
    with open(BASE_PATH+LISA_CSV, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        for row in spamreader:
            if not head:
                k = row[0].rfind("/") + 1
                filename = row[0][k:len(row[0])].replace('.png', '.xml')

                if filename != prev_file:
                    tree = build_xml_tree(row)
                else:
                    in_file = open(ANNOTATIONS_OUTPUT_DIR + filename)
                    tree = ET.parse(in_file)
                    root = tree.getroot()
                    tree = new_object(root, row)

                tree.write(ANNOTATIONS_OUTPUT_DIR + filename)
                print("processed: "+filename)
                prev_file = filename
            else:  # Skip column def
                head = False


if __name__ == '__main__':
    main()


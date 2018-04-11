#!/envs/drAIver/bin/python
import csv
import xml.etree.cElementTree as ET

LISA_CSV = "Datasets/drAIver/LISA/annotations/vid_annotations_test.csv"
# BASE_PATH = "/mnt/B01EEC811EEC41C8/" # Ubuntu Config
BASE_PATH = "/Users/marco/Documents/"


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

    object = ET.SubElement(root, "object")

    ET.SubElement(object, "name").text = row[1]

    bndbox = ET.SubElement(object, "bndbox")

    ET.SubElement(bndbox, "xmin").text = row[2]
    ET.SubElement(bndbox, "ymin").text = row[3]
    ET.SubElement(bndbox, "xmax").text = row[4]
    ET.SubElement(bndbox, "ymax").text = row[5]

    tree = ET.ElementTree(root)
    return tree

# def build_xml_tree_fake():
#     root = ET.Element("annotation")
#
#     ET.SubElement(root, "filename").text = "Filename"
#
#
#
#     object = ET.SubElement(root, "object")
#
#     ET.SubElement(object, "name").text = "Name"
#
#     bndbox = ET.SubElement(object, "bndbox")
#
#     ET.SubElement(bndbox, "xmin").text = "12"
#     ET.SubElement(bndbox, "ymin").text = "13"
#     ET.SubElement(bndbox, "xmax").text = "14"
#     ET.SubElement(bndbox, "ymax").text = "15"
#
#     tree = ET.ElementTree(root)
#     return tree


def main():
    # tree = build_xml_tree_fake()
    # tree.write("test.xml", pretty_print=True)

    head = True
    with open(BASE_PATH+LISA_CSV, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        for row in spamreader:
            if not head:
                k = row[0].rfind("/") + 1
                filename = row[0][k:len(row[0])].replace('.png', '.xml')

                tree = build_xml_tree(row)
                tree.write(filename)

                print(row)
            else:  # Skip column def
                head = False


if __name__ == '__main__':
    main()


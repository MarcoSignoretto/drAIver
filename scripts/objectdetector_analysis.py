#!/envs/drAIver/bin/python
import _pickle as pk
import sys, getopt


def main(option):
    with open('report_%s.pickle' % option, 'rb') as handle:
        report = pk.load(handle)

def compute_mAP(report):
    pass

def precision_recall_curve(object_class):
    pass


if __name__ == '__main__':

    option = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:', ['dataset='])
    except getopt.GetoptError:
        print('objectdetector_evaluation.py -d <kitty/lisa>')
        sys.exit(2)

    for o, a in opts:
        if o in ("-d", "--dataset"):
            if a == "kitty":
                option = a
            elif a == "lisa":
                option = a
        else:
            assert False, "unhandled option"

    main(option)



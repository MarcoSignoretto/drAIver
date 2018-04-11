# Object detector training

In order to train object detector on custom datasets some adjustments are required.

All the following works must be done in the ```darkflow```installation folder.

In particular we need to adjust the ```labels.txt``` and the ```cfg```file to adapt to our dataset.

## LISA Dataset training

The config file for LISA dataset is available [here](https://github.com/MarcoSignoretto/drAIver/tree/master/data/trafficsigns_detector/tiny-yolov2-lisa.cfg) and the labels files are available [here](https://github.com/MarcoSignoretto/drAIver/tree/master/data/trafficsigns_detector/labels.txt) 

1. Convert LISA annotations to Darkflow annotations using the script in ```scripts/LISA_to_darknet.py``` // TODO complete
2. Put ```tiny-yolov2-lisa.cfg``` into the ```<darkflow_home>/cfg/``` folder
3. Put ```labels.txt``` into the ```<darkflow_home>``` folder
4. copy the ```annotation_folder``` into ```<darkflow_home>/training/lisa/annotations``` // TODO complete
5. copy the ```image_folder``` into ```<darkflow_home>/training/lisa/images``` // TODO complete
6. Start training executing the collowing commands from the ```<darkflow_home>``` folder.

```sh
flow --model cfg/tiny-yolov2-lisa.cfg --train --dataset "training/lisa/images" --annotation "training/lisa/annotations" --labels "training/lisa/labels.txt"
```
6. Save the training result into ```.pb``` file 
```sh
flow --model cfg/tiny-yolo-new.cfg --load -1 --savepb
```
// TODO continue from here
## Lisa testing
```sh
flow --model "test/training/sample_lisa/cfg/tiny-yolov2-lisa-test.cfg" --train --dataset "test/training/sample_lisa/images" --annotation "test/training/sample_lisa/annotations" --labels "test/training/sample_lisa/labels.txt"
```


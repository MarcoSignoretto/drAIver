# Object detector training

In order to train object detector on custom datasets some adjustments are required.

All the following works must be done in the ```darkflow```installation folder.

In particular we need to adjust the ```labels.txt``` and the ```cfg```file to adapt to our dataset.

## LISA Dataset training

The config file for LISA dataset is available [here](https://github.com/MarcoSignoretto/drAIver/tree/master/data/trafficsigns_detector/tiny-yolov2-lisa.cfg) and the labels files are available [here](https://github.com/MarcoSignoretto/drAIver/tree/master/data/trafficsigns_detector/labels.txt) 

1. create ```annotations``` directory into the ```lisa``` folder
2. Convert LISA annotations to Darkflow annotations using the script in ```scripts/LISA_to_darknet.py``` and set as output the folder above
3. Copy all the contents of the ```data/lisa``` into ```<darkflow_home>/training```
4. create ``images``` directory into the ```lisa``` folder and copy all the vid files inside it
5. Start training executing the following commands from the ```<darkflow_home>``` folder.

// TODO use also gray scale images???

```sh
source activate drAIver
flow --model "training/lisa/cfg/tiny-yolov2-lisa.cfg" --train --dataset "training/lisa/images" --annotation "training/lisa/annotations" --labels "training/lisa/labels.txt"
```
6. Save the training result into ```.pb``` file 
```sh
flow --model "training/lisa/cfg/tiny-yolov2-lisa.cfg" --load -1 --savepb
```
// TODO continue from here
## Lisa testing
```sh
flow --model "training/sample_lisa/cfg/tiny-yolov2-lisa-test.cfg" --train --dataset "training/sample_lisa/images" --annotation "training/sample_lisa/annotations" --labels "training/sample_lisa/labels.txt"
```


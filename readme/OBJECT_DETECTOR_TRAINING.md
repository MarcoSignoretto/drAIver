# Object detectors training

In order to train object detector on custom datasets some adjustments are required.

All the following works must be done in the ```darkflow```installation folder.

In particular we need to adjust the ```labels.txt``` and the ```cfg```file to adapt to our dataset.

## LISA

### Training

The config file for LISA dataset is available [here](https://github.com/MarcoSignoretto/drAIver/tree/master/data/trafficsigns_detector/tiny-yolov2-lisa.cfg) and the labels files are available [here](https://github.com/MarcoSignoretto/drAIver/tree/master/data/trafficsigns_detector/labels.txt) 

1. create ```annotations``` directory into the ```lisa``` folder
2. Convert LISA annotations to Darkflow annotations using the script in ```scripts/LISA_to_darknet.py``` and set as output the folder above
3. Copy all the contents of the ```data/lisa``` into ```<darkflow_home>/training/lisa```
4. create ``images``` directory into the ```lisa``` folder and copy all the vid files inside it
5. Start training executing the following commands from the ```<darkflow_home>``` folder.

// TODO use also gray scale images???

```sh
source activate drAIver
flow --model "training/lisa/cfg/tiny-yolov2-lisa.cfg" --train --dataset "training/lisa/images" --annotation "training/lisa/annotations" --labels "training/lisa/labels.txt"
```
6. Save the training result into ```.pb``` file 
```sh
flow --model "training/lisa/cfg/tiny-yolov2-lisa.cfg" --load -1 --labels "training/lisa/labels.txt" --savepb
```
// TODO continue from here
### Testing
```sh
flow --model "training/sample_lisa/cfg/tiny-yolov2-lisa-test.cfg" --train --dataset "training/sample_lisa/images" --annotation "training/sample_lisa/annotations" --labels "training/sample_lisa/labels.txt"
```

## KITTY

### Training

Since the format of darkflow is the same of the VOC dataset there is a repository with a script that are able to convert the KITTY dataset annotation format to the darkflow one

In order to train the network on KITTY execute the following commands

1. Clone the repo available [here](https://github.com/umautobots/vod-converter)
2. Put in ```<KITTY_HOME>/training/images_2``` the Kitty images
3. Put in ```<KITTY_HOME>/training/label_2``` the Kitty labels for the above images
4. Copy the file ```train.txt``` that is in ```data/kitty``` into ```<KITTY_HOME>```
5. Run the conversion script with the following commands:
```sh
cd voc-converter
python3.6 vod_converter/main.py --from kitti --from-path <KITTY_HOME> --to voc --to-path <KITTY_HOME>/darkflow
```
3. Copy all the contents of the ```data/kitty``` into ```<darkflow_home>/training/kitty```
6. Copy all the contents of ```images_2```  into ```<darkflow_home>/training/images```
7. Copy all the contents of ```darkflow/VOC2012/Annotations```  into ```<darkflow_home>/training/annotations```
8. execute the following commands to train the network

```sh
source activate drAIver
flow --model "training/kitty/cfg/tiny-yolov2-kitty.cfg" --train --dataset "training/kitty/images" --annotation "training/kitty/annotations" --labels "training/kitty/labels.txt"
```

9. Save the training result into ```.pb``` file
```sh
flow --model "training/kitty/cfg/tiny-yolov2-kitty.cfg" --load -1 --labels "training/kitty/labels.txt" --savepb
```

### Testing

// TODO add command to load network from .pb

# Training using Darknet 

// TODO remember to change training to testing when trsining is completed

1. Copy ```cfg/kitty.data``` into darknet ```cfg``` folder
2. Copy ```cfg/tiny-yolov3-kitty.cfg``` into darknet ```cfg``` folder
3. Copy ```data/kitty.names``` into darknet ```data``` folder
```sh
./darknet detector train cfg/kitty.data cfg/tiny-yolov3-kitty.cfg darknet53.conv.74
```


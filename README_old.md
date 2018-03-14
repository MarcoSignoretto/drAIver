OLD stuffs here


#### Install pip packages ####
```sh
workon cv
pip install matplotlib
pip install simplejson #used for points cloud reconstruction
```

#### Install StereoVision library ####

```sh
git clone git@github.com:MarcoSignoretto/StereoVision.git
cd StereoVision
python setup.py install
cd ..
rm -rf progressbar-python3
```

issues with python3 opencv3 https://github.com/erget/StereoVision/issues/13

forked for python3 openCv3

copy bin and stereovision in correct position on virtual env
modify point_clouds.py,
use my version

#### Install progress bar python3 ####

```sh
git clone https://github.com/coagulant/progressbar-python3.git
cd progressbar-python3
python setup.py install
cd ..
rm -rf progressbar-python3
```

#### Install MeshLab ####

Install MeshLab to see points cloud

## Install AccessPoint ##

WARNING Used standalone ubuntu 16.04 Hotspot creation



## Install Open Kinect ##

Official reference: https://openkinect.org/wiki/Getting_Started/it

Follow tutorial for python usage: https://naman5.wordpress.com/2014/06/24/experimenting-with-kinect-using-opencv-python-and-open-kinect-libfreenect/

### Ubuntu 16.04 ###
```sh
sudo add-apt-repository ppa:arne-alamut/freenect ( Lib freenect )
sudo apt-get update
sudo apt-get install freenect
sudo adduser TUONOME video
```

### Mac Osx Sierra ( 12.12 ) ###
```sh
brew install libfreenect
```



# Usage #
Use the virtual env ```cv```
to congigure virtual env in PyCharm see http://exponential.io/blog/2015/02/10/configure-pycharm-to-use-virtualenv/

## using StereoVision library for Calibration ##
github: https://github.com/erget/StereoVision
In order to execute all this commands ensure to be in ```cv``` virtual env

1. Shows cameras
```sh
show_webcams 1 2
```

2. Take calibration photos
```sh
capture_chessboards 1 2 50 calibration_pictures/  # warning photos are taken only if marker is recognized
```

3. Calibration phase
```sh
time calibrate_cameras --rows 9 --columns 6 --square-size 1.8 calibration_pictures/ calibration/
```
Note ( python2 print function rename print line 50 with python3 print: print())

4. reconstruction
```sh
images_to_pointcloud calibration/ reconstruction/left_10.ppm reconstruction/right_10.ppm reconstruction/reconstruction_10.ply
```
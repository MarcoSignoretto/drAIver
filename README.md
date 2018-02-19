# DrAIver project #

DrAIver is a self driving robot ables to recognize pedestriands and street signals while it explores the world.- 

# Connection to Raspberry Pi 3 #

In order to connect your lapto to your raspberry execute the following commands:



```sh
	ssh pi@dex.local
```

password: robot1234

# Setup #

## Raspberry Pi 3 Model B Setup ##

For this project will be used Python 3.4 with OpenCv 3.2.0 and tensorflow 1.1

### SDCard Setup ###

1. Use gparted of Ubuntu to partitioned SDCard ( sing FAT partition ) 
2. Install Raspbian for Robots ( Download it from Dexter Industries website )
3. Format sdcard as FAT32 ( Use gparted )
4. Using Etcher to flash Image into SDCard

### Prepare for Robot ###

1. Enable ssh connection

```sh
	sudo raspi-config
```

Interfacing Options > SSH > Yes

2. Enable VNC Server

```sh
	sudo raspi-config
```

Interfacing Options > VNC > Yes
see also for other references https://www.raspberrypi.org/documentation/remote-access/vnc/

### BrickPi Setup ( OLD ) ###

```sh
git clone https://github.com/DexterInd/BrickPi.git
cd BrickPi
cd Setup Files
sudo chmod +x install.sh
sudo ./install.sh
```

references : https://www.dexterindustries.com/BrickPi/brickpi-tutorials-documentation/getting-started/pi-prep/



## Install OpenCv3 with Python3 ##

instructions: http://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/

```sh
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.2.0/modules \
    -D BUILD_EXAMPLES=ON ..
```

## Install TensorFlow 1.1 ##

Install tensorflow for Raspberry Pi 3

references: https://github.com/samjabrahams/tensorflow-on-raspberry-pi

### Bazel ( OLD ) ###

No change to arm ( remember to change permissions if needs some edit )

### Compile TensorFlow ( OLD ) ###

references: https://www.tensorflow.org/install/install_sources ( Official docs + doc below)

sudo nano tensorflow/core/platform/platform.

Comment if !raspberry to avoid problems
```sh
// Require an outside macro to tell us if we're building for Raspberry Pi.
//#if !defined(RASPBERRY_PI)
//#define IS_MOBILE_PLATFORM
//#endif  // !defined(RASPBERRY_PI)
```

```sh
./configure
bazel build -c opt --copt="-mfpu=neon-vfpv4" --copt="-funsafe-math-optimizations" --copt="-ftree-vectorize" --copt="-fomit-frame-pointer" --local_resources 1024,1.0,1.0 --verbose_failures tensorflow/tools/pip_package:build_pip_package --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
```


bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --local_resources 1024,1.0,1.0

### Ubuntu 16.04 ###

Tutorial: http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/

cmake configuration used:
```sh
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.2.0/modules \
    -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
    -D BUILD_EXAMPLES=ON ..

instead of ```make -j4``` use ```make``` to avoid compilation problems

remember to execute:
```sh
sudo make install
sudo ldconfig
```

### Mac Osx Sierra ( 12.12 ) ###

Tutorial: http://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5/

#### Installing python3 from Homebrew ####

```sh
brew install python3
```

WARNING remember to type following command to check correct settings:
```sh
brew link --overwrite python3
brew linkapps python3
```
Then execute the followig commands:
```sh
python3
from distutils.sysconfig import get_python_inc; print(get_python_inc())
```
you look if output is similar to:
```sh
/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/include/python3.6m
```

#### Compile OpenCv ####

Compile options:
```sh
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.2.0/modules \
    -D PYTHON3_LIBRARY=/usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/lib/python3.6/config-3.6m-darwin/libpython3.6.dylib \
    -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D BUILD_EXAMPLES=ON ..
```

WARNING remember to change all python 3.5 with 3.6!!!!!

### Windows 10 ###

Tutorial: https://www.solarianprogrammer.com/2016/09/17/install-opencv-3-with-python-3-on-windows/

### All platforms ###

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
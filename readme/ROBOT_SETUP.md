
# Robot environment setup #

The robot environment are based on Python 3.4 with OpenCv 3.3.0

## Prepare the SD card ##

1. Use gparted of Ubuntu to partitioned SDCard
2. Format the sdcard as FAT32 ( Use gparted )

## Installing Raspbian for Robots ##

First of all we need to install the Operating System on our RaspberryPi3, to do that we need to follow the [Official Installation Guide from Dexter Industries](https://www.dexterindustries.com/howto/install-raspbian-for-robots-image-on-an-sd-card/)

// TODO add my image base + completed


## Raspberry setup ##

1. Expands filesystem
```sh
sudo raspi-config
```

Expand filesystem > Enter > Reboot the raspberry with ```sudo reboot```

2. Enable ssh connection

```sh
	sudo raspi-config
```

Interfacing Options > SSH > Yes

3. Enable VNC Server

```sh
	sudo raspi-config
```

Interfacing Options > VNC > Yes
see also for [Official reference](https://www.raspberrypi.org/documentation/remote-access/vnc/) for further informations.

## Create drAIver virtual env ##

Some setup are required to create virtual envs

### Install dependencies ###

```sh
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo pip install virtualenv virtualenvwrapper
sudo rm -rf ~/.cache/pip

echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.profile
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.profile
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.profile

source ~/.profile # remember to execute at each terminal opening otherwise only ssh connection run ~/.profile
```

### Create drAIver virtual env ###

```sh
mkvirtualenv cv -p python3
workon drAIver # if terminal prepend (drAIver) all is ok
```

### Link envs ###

In order to run scripts without problems we need to create a symbolic link from ```envs/``` to ```~/.virtualenvs/```

```bash
cd /
sudo ln -s ~/.virtualenvs/ envs
```

### Link drAIver scripts as library ###

In order to use drAIver scripts as library you need to import it into the ```site-packages``` folder, to do that execute the following lines of code

```bash
cd /envs/drAIver/lib/python3.4/site-packages/
sudo ln -s <drAIver_folder>/draiver/ draiver
```

## Install OpenCv3 with Python3 ##

### Install dependencies ###

1. Execute the following commands to install dependencies

```sh
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python2.7-dev python3-dev
```

2. install numpy on virtual env
```sh
workon drAIver
pip install numpy
```

### Download OpenCv sources ###

Clone the source from the official GitHub repo

```sh
cd ~
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 3.3.0
cd ..
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 3.3.0
cd ..
```



### Compile OpenCv ###

1. Prepare the CMake
```sh
cd ~/opencv/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..
```

2. Compile ( takes very long time )
```sh
make # avoid to use -j because OpenCv failed on compilation with multiple threads
sudo make install
sudo ldconfig
```

### Setup OpenCv Python bindings ###

```sh
cd /usr/local/lib/python3.4/site-packages/
sudo mv cv2.cpython-34m.so cv2.so

cd ~/.virtualenvs/cv/lib/python3.4/site-packages/
ln -s /usr/local/lib/python3.4/site-packages/cv2.so cv2.so
```

### Testing the installation ###

Open new terminal and execute

```sh
source ~/.profile 
workon drAIver
python
>>> import cv2
>>> cv2.__version__
'3.3.0'
>>>
```

## Install BrickPi3 library on drAIver virtual env ##

In order to install BrickPi library on our virtual env we need to execute the following commands:

```sh
workon drAIver
git clone https://github.com/DexterInd/BrickPi3.git
cd <library_directory>/BrickPi3/Software/Python
python setup.py clean
python setup.py build
python setup.py install
pip install spidev
```

### Test the installation ###

From drAIver virtual env open python and execute:

```python
import BrickPi3
```

If no error your setup is correct.

## Hotspot setup ##

In order to setup the hotspot on your raspberryPi follow the [official guide](https://www.raspberrypi.org/documentation/configuration/wireless/access-point.md) but set as SSID ```drAIver```.

## Samba shared folder ##

In order to easy share code between computer and raspberry is usefull to setup a samba server, the raspberry image already has it installed so we must only configure it.

In order to do that we need to execute the following commands:

```sh
mkdir ~/share
sudo nano /etc/samba/smb.conf
```

at the bottom of the file add these lines:
```sh
[drAIverShare]
   comment=drAIver shared folder
   path=/home/pi/share
   browseable=Yes
   writeable=Yes
   only guest=no
   create mask=0777
   directory mask=0777
   public=no
```
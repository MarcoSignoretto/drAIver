# Connection to Raspberry Pi 3 #

In order to connect your laptop to your raspberry execute the following commands:

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




# TODO complete with hotspot setup #
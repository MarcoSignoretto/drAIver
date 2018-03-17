# DrAIver #

# WORK IN PROGRESS!!!!

In order to use the drAIver robot we need to perform some setup steps.

## Setup ##

### Install PC environment ###

To setup the PC or Computer environments follow the instructions [here](https://github.com/MarcoSignoretto/drAIver/blob/master/readme/ENVIRONMENT_SETUP.md).

### Robot setup ###

To prepare and configure the robot follow the instructions [here](https://github.com/MarcoSignoretto/drAIver/blob/master/readme/ROBOT_SETUP.md)

## Usage ##

### Connect to the robot ###

1. Turn on the robot
2. Choose the wireless connection with SSID ```drAIver``` ( some times is required to robot startup so wait )
3. Execute the following commands to establish ssh connection ( the password is: ```robots1234``` )
```sh
	ssh pi@drAIver.local
```

### Work on drAIver virtual environment ###

1. execute ```lsvirtualenv``` to show all available virtual envs
2. activate drAiver with ```workon drAIver```

### Connect to the Samba shared folder ###

The parameters to connect to the shared folder are: ```smb://pi@drAIver.local``` and prompt the password ```robots1234```

## References ##

[BrickPi3 Documentation](http://www.aplu.ch/classdoc/dexter/brickpi3.BrickPi3-class.html)

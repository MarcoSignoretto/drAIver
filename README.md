# DrAIver #

## Demo videos ##

### Road following:

[External view](https://drive.google.com/file/d/1tIYad0tsVv1Fd4LVUhAb8e7hwKqH_dV1/view?usp=sharing)

[Robot view](https://drive.google.com/file/d/1rOc9yYvLTeFOjG5vaNjxqSVVQSCD0K2V/view?usp=sharing)

### Car detection:

[External view](https://drive.google.com/file/d/1cAShFMxX35htLiZkBvK1kBUiuEWXJQzj/view?usp=sharing)

[Robot view](https://drive.google.com/file/d/1MmRhEAC2cFX-3oko_xAf0Hx4lZSznT1y/view?usp=sharing)


### Pedestrian crossing detection:

[External view](https://drive.google.com/file/d/1NjWoQy-oa3D3TN-WdIjFJu-TYr1nldgX/view?usp=sharing)

[Robot view](https://drive.google.com/file/d/13B1uoI9_kUGv8qaAIvRhAwcQE36H24Up/view?usp=sharing)

### Stop detection:

[External view](https://drive.google.com/file/d/1hiPH3ikfg2NBJ2Lf92d1gT4ozM0t-bHy/view?usp=sharing)

[Robot view](https://drive.google.com/file/d/1UsOeMyfGYTlytQnVmom1pPP_58jx0-wy/view?usp=sharing)

In order to use the drAIver robot we need to perform some setup steps.

## Setup ##

### Install PC environment ###

To setup the PC or Computer environments follow the instructions [here](https://github.com/MarcoSignoretto/drAIver/blob/master/readme/ENVIRONMENT_SETUP.md).

### Robot setup ###

To prepare and configure the robot follow the instructions [here](https://github.com/MarcoSignoretto/drAIver/blob/master/readme/ROBOT_SETUP.md)

## Usage ##

### Connect to the robot ###

1. Turn on the robot
2. Choose the wireless connection with SSID ```drAIver``` ( startup may be long so wait )
3. Execute the following commands to establish ssh connection ( the password is: ```robots1234``` )
```sh
	ssh pi@drAIver.local
```

### Work on drAIver virtual environment ###

1. execute ```lsvirtualenv``` to show all available virtual envs
2. activate drAIver with ```workon drAIver```

### Connect to the Samba shared folder ###

The parameters to connect to the shared folder are: ```smb://pi@drAIver.local``` and prompt the password ```robots1234```

## Train detector on custom Dataset

To the training procedure see the section [training](https://github.com/MarcoSignoretto/drAIver/blob/master/readme/OBJECT_DETECTOR_TRAINING.md).

## References ##

[BrickPi3 Documentation](http://www.aplu.ch/classdoc/dexter/brickpi3.BrickPi3-class.html)

[Thesis](https://github.com/MarcoSignoretto/drAIver/blob/master/readme/Signoretto_Marco_drAIver.pdf)
# Desktop Environment setup #

All commands are related to root directory of the project

## Setup WINDOWS ##
1. Install anaconda 3
2. Open terminal and execute:  ```conda env create -f setup/env/drAiver_pc_env.yml```

## Setup Mac ##
1. Install anaconda 3
2. Open terminal and execute:  ```conda env create -f setup/env/drAiver_mac_env.yml```

4. execute ln -s on tensorflow-models/research and  tensorflow-models/research/slim

### Link tensorflow library into python env ###
1. Download [tensorflow-models]()
2. Compile protobuf with
```sh
# From tensorflow-models/research/
protoc object_detection/protos/*.proto --python_out=.
 ```
3. link libraries to current virtual env
```sh
ln -s <<PATH-TO-TENSORFLOW-MODEL>>tensorflow-models/research/* <<ANACONDA-INSTALLATION-DIR>>/anaconda3/envs/drAIver/lib/python3.5/site-packages/
ln -s <<PATH-TO-TENSORFLOW-MODEL>>/tensorflow-models/research/slim/* <<ANACONDA-INSTALLATION-DIR>>/anaconda3/envs/drAIver/lib/python3.5/site-packages/
 ```

 ### not definitive ###
 [Install darkflow](https://github.com/thtrieu/darkflow)

### windows installation ###
1. for windows copy rc.exe e rcdll.dll to visual studio bin directory
2. create shorcut to darkflow from site-packages of the virtual env
3. copy flow outsite the original folder otherwise it doesn't work ( example create a folder flowdir and put it inside)


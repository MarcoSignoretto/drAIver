# Desktop Environment setup #

All commands are related to root directory of the project

## Setup WINDOWS ##
1. Install anaconda 3
2. Open terminal and execute:  ```conda env create -f setup/env/drAiver_pc_env.yml```

## Setup Mac ##
1. Install anaconda 3
2. Open terminal and execute:  ```conda env create -f setup/env/drAiver_pc_env.yml```

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
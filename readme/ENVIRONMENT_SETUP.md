# Desktop Environment setup #

## NVIDIA Cuda ##

We need to install the NVIDIA Toolkit 8.0 and cuDNN 6.0,
this is because we will use tensorflow 1.4.0

To setup cuda follow the official documentations

1. [Cuda toolkit installation](http://developer.download.nvidia.com/compute/cuda/8.0/secure/Prod2/docs/sidebar/CUDA_Quick_Start_Guide.pdf?jt8hlC2y6mZQpU3FGuS8yF7Lj0fRVC6LBxKROaolCbk4cbeQmp6QJk9cTRaPL1V_bAHkdem4lJoRgCJIl4mr7DAd7qgn4WHkqIU1VQrY6V501GTlncn9ySo8k3VAX-yxU6NRGjAkt-dFiAJJxO70PgmX9QrswHfYRfMETOgLMZWEuFIe)
2. [cuDNN Installation](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installdriver)

WARNING For ubuntu installation:
1. ensure to execute ```sudo apt-get install cuda-8-0``` to install Toolkit 8.0 otherwise the newest one will be installed.
2. ensure to add 
```sh
# NVIDIA CUDA
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME="$CUDA_HOME:/usr/local/cuda"
```
to your ```~/.bashrc``` and then ```source ~/.bashrc```

## Anaconda ##

All commands are related to root directory of the project

1. Install anaconda 3 for the correct operating system [Anaconda3 Download](https://www.anaconda.com/download/)
2. choose your correct environment, the environments are located into the ```setup/env``` folder and the environemnt file has the following name structure:  ```drAIver_<operation_system>[_gpu]_env.yml``` where ```<operation_system>``` could be ```ubuntu```, ```osx```, ```windows```. ```[_gpu]``` is optional, if present the configuration include ```Tensorflow-gpu``` instead of ```Tensorflow```

The environment file that you choose will be called ```<your_environment>```.

WARNING: not all the possible environment combinations are present on ```setup/env``` the folder.

### Create the environment ###
Open terminal and execute:  ```conda env create -f setup/env/<your_environment>```

### Link anaconda envs ###

In order to run scripts without problems we need to create a symbolic link from ```/envs``` to ```~/anaconda3/envs```

```bash
cd /
sudo ln -s ~/anaconda3/envs/ envs
```

### Link drAIver scripts as library ###

In order to use drAIver/driver scripts as library you need to import it into the ```site-packages``` folder, to do that execute the following lines of code

```bash
cd /envs/drAIver/lib/python3.5/site-packages/
sudo ln -s <drAIver_folder>/draiver/ draiver
```


#### Testing installation ####
1. ```source activate drAIver```
2. execute script
```python
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()
```

### Link tensorflow-models library into python env ###
1. Clone [tensorflow-models](https://github.com/tensorflow/models)
```sh
git clone https://github.com/tensorflow/models.git tensorflow-models
 ```

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

### Install darkflow (Deprecated) ###
 [Install darkflow](https://github.com/thtrieu/darkflow)

1. Install the library
 ```sh
git clone https://github.com/thtrieu/darkflow.git darflow
source activate drAIver
pip install Cython
cd darkflow
pip install .
mkdir bin
 ```

### Install Darknet for drAIver ###

Clone darknet from my fork [Darknet for drAIver](https://github.com/MarcoSignoretto/darknet)
and follow the instructions of the official repo see below, remember to change the Makefile

#### Python binding ####

1. Go to Darknet folder
2. Open ```python/darknet.py```
3. Changing the CDLL path with the path of the ```libdarknet.so``` (example /Users/marco/Documents/GitProjects/UNIVE/darknet/libdarknet.so)
4. Copy lib file into correct folder
```sh
cp python/darknet.py /envs/drAIver/lib/python3.5/darknet.py
```
5. Adjust the ```.data``` file with absolute paths
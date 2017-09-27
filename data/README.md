## Data Generation Script for Real Env

### Requirements
Recommend: create virtual environment for the dependencies
Install Blender 2.79
```shell
sudo add-apt-repository ppa:thomas-schiex/blender
sudo apt-get update
sudo apt-get install blender
```
Create necessary environment
```shell
conda create -n (env_name) python=3.5
pip install pillow
curl -sL https://deb.nodesource.com/setup_8.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install optimist bytebuffer long
```


### Run the code
```shell
source activate (env_name)
python start.py
```

## Physics Simulation Engine for Real Env
Zhiyang He

### Requirements
Python 3 environment (technically pybullet builds on python2, but I haven't been able to get it working)
```
Create necessary environment
```shell
conda create -n (env_name) python=3.5
pip install pybullet, zmq
```

Make proper obj file for OpenGL convention
```
python invert_obj --datapath ../data --model model_id
```


### Run the code
```shell
python render_physics.py --datapath ../data --model model_id
```
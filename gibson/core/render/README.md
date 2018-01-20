## Real Universe Viewer Renderer

Renderer for real-time visual input. Currently neural network filler is not included, rendering framerate: 30fps in single view, 12fps with five views.

Beta version renderer is implemented using OpenCV-Python. In the future we might completely switch to OpenGL to avoid the speed overhead, and reduce setup cost.

Install opencv, pytorch and others
```shell
conda install -c menpo opencv -y
conda install pytorch torchvision cuda80 -c soumith
conda install pillow
pip install zmq
```

Build cpp, cython dependencies
```shell
bash build.sh
bash build_cuda.sh
python setup.py build_ext --inplace
```

Download the necessary helper files
```shell
wget https://www.dropbox.com/s/a4tb959bbue0hap/coord.npy
```

```
source activate (py2_universe_env)
python show_3d2.py --datapath ../data/ --idx 10
```
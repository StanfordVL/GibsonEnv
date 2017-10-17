#!/bin/bash
cd ./realenv/core/render/
bash build.sh
bash build_cuda.sh
python setup.py build_ext --inplace
cd -
make ./realenv/core/channels/build
cd ./realenv/core/channels/build
cmake .. & make -j 10
cd -
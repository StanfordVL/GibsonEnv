#!/bin/bash
g++ -std=c++11 ./realenv/envs/render.cpp -o ./realenv/envs/render.so -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0



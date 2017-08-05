#!/bin/bash
g++ -std=c++11 render.cpp -o render.so -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0



#!/bin/bash
g++ -std=c++11 -fopenmp -Wall -c render.cpp -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0
g++ -shared -lgomp -lrt -o render.so render.o

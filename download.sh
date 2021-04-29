#!/bin/bash
if [ `basename "$PWD"` = GibsonEnv ]; then
    if [ ! -d /tmp/gibson ]; then
        mkdir /tmp/gibson
    fi
    wget https://storage.googleapis.com/gibsonassets/assets_core_v2.tar.gz -O /tmp/gibson/assets_core_v2.tar.gz
    wget https://storage.googleapis.com/gibsonassets/dataset.tar.gz -O /tmp/gibson/dataset.tar.gz
    tar -zxf /tmp/gibson/assets_core_v2.tar.gz -C $PWD/gibson
    tar -zxf /tmp/gibson/dataset.tar.gz -C $PWD/gibson/assets/
else
    echo "please run in GibsonEnv directory"
fi

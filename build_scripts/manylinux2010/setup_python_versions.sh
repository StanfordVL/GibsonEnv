#!/bin/bash

# Upgrade the system
yum check-update
yum update

yum groupinstall -y "Development Tools"

yum install -y gcc bzip2-devel sqlite-devel zlib-devel wget libffi-devel

declare -A python_versions=(
  ['3.6']='https://www.python.org/ftp/python/3.6.4/Python-3.6.4.tar.xz'
  ['3.7']='https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tar.xz'
  ['3.8']='https://www.python.org/ftp/python/3.8.9/Python-3.8.9.tar.xz'
  #['3.9']='https://www.python.org/ftp/python/3.9.4/Python-3.9.4.tar.xz'
)

# Install openssl
if [ $python_version = "3.6" ]; then
  yum install -y openssl-devel
else
  git clone https://github.com/openssl/openssl.git
  cd openssl
  git checkout OpenSSL_1_1_1-stable
  git submodule update --init
  ./config
  make -j2
  make install
  cd ..
fi

# Build and install python
file_name=$(basename ${python_versions[$python_version]})
dir_name=${file_name%.*.*}

wget ${python_versions[$python_version]}
tar -xf $file_name
cd $dir_name

./configure
make -j2
make install

python3 -m pip install --upgrade pip
python3 --version
pip3 --version
cd ..
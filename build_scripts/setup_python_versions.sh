#!/bin/bash

yum groupinstall -y "Development Tools"
yum install -y gcc openssl-devel bzip2-devel sqlite-devel zlib-devel wget
declare -A python_versions=(
  ['3.6']='https://www.python.org/ftp/python/3.6.4/Python-3.6.4.tar.xz'
  ['3.7']='https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tar.xz'
  ['3.8']='https://www.python.org/ftp/python/3.8.4/Python-3.8.4.tar.xz'
  ['3.9']='https://www.python.org/ftp/python/3.9.4/Python-3.9.4.tar.xz'
)

for version in "${!python_versions[@]}"; do
  file_name=$(basename ${python_versions[$version]})
  dir_name=${file_name%.*.*}

  wget ${python_versions[$version]}
  tar -xf $file_name

  cd $dir_name

  ./configure
  make
  make install

  python$version -m pip install --upgrade pip
  python$version --version
  pip$version --version
  cd ..
done






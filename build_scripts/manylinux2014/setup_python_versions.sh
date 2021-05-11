#!/bin/bash

function install_python() {
  file_name=$(basename ${python_versions[$1]})
  dir_name=${file_name%.*.*}

  wget ${python_versions[$1]}
  tar -xf $file_name
  cd $dir_name

  ./configure
  make -j2
  make install

  python$1 -m pip install --upgrade pip
  python$1 --version
  pip$1 --version
  cd ..

  rm -rf $file_name
}

yum install -y gcc bzip2-devel sqlite-devel zlib-devel wget libffi-devel openssl-devel

declare -A python_versions=(
  ['3.6']='https://www.python.org/ftp/python/3.6.4/Python-3.6.4.tar.xz'
  ['3.7']='https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tar.xz'
  ['3.8']='https://www.python.org/ftp/python/3.8.9/Python-3.8.9.tar.xz'
  ['3.9']='https://www.python.org/ftp/python/3.9.4/Python-3.9.4.tar.xz'
)

# Install python 3.6
install_python "3.6"

yum remove -y openssl-devel

# Install openssl to build new python versions
git clone https://github.com/openssl/openssl.git
cd openssl
git checkout OpenSSL_1_1_1-stable
git submodule update --init
./config
make -j2
make install
cd ..

for version in "${!python_versions[@]}"; do
  if [ $version != "3.6" ]; then
    install_python $version
  fi
done

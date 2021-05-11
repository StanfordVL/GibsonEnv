# vi /var/tmp/test_script.sh
#!/bin/bash

# Install pip requirements
export CC=/usr/lib64/openmpi/bin/mpicc

declare -A python_versions=(
  ['3.6']='https://www.python.org/ftp/python/3.6.4/Python-3.6.4.tar.xz'
  ['3.7']='https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tar.xz'
  ['3.8']='https://www.python.org/ftp/python/3.8.9/Python-3.8.9.tar.xz'
  #['3.9']='https://www.python.org/ftp/python/3.9.4/Python-3.9.4.tar.xz'
)

for version in "${!python_versions[@]}"; do
  pip$version install wheel auditwheel
  pip$version install -r build_scripts/manylinux2010/requirements.txt --no-cache-dir
  pip$version install -e .
  python$version setup.py bdist_wheel
  python$version -m auditwheel repair dist/gibson-*-cp"${version//.}"*
  pip$version uninstall -r build_scripts/manylinux2010/requirements.txt -y
  rm -rf /tmp/pip*
done


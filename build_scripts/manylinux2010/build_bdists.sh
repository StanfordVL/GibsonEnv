# vi /var/tmp/test_script.sh
#!/bin/bash

# Install pip requirements
export CC=/usr/lib64/openmpi/bin/mpicc

pip3 install wheel auditwheel
pip3 install -r build_scripts/manylinux2010/requirements.txt
pip3 install -e .
python3 setup.py bdist_wheel
python3 -m auditwheel repair dist/gibson-*-cp*



# vi /var/tmp/test_script.sh
#!/bin/bash

# Upgrade the system
yum check-update

# Install dependency for mpi4pi
yum install openmpi-devel -y
export CC=/usr/lib64/openmpi/bin/mpicc

# Install python requirements
pip install -r requirements.txt
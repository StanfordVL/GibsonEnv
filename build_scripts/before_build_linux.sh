# Setup CUDA
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CC=/usr/lib64/openmpi/bin/mpicc

# Install pip requirements
pip install -r build_scripts/requirements.txt
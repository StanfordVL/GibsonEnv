FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04


#RUN echo -e "\n**********************\nNVIDIA Driver Version\n**********************\n" && \
#	cat /proc/driver/nvidia/version && \
#	echo -e "\n**********************\nCUDA Version\n**********************\n" && \
#	nvcc -V && \

## Skip keyboard settings
ENV DEBIAN_FRONTEND noninteractive

# Install some dependencies
RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		gfortran \
		git \
		libffi-dev \
		libfreetype6-dev \
		libhdf5-dev \
		libjpeg-dev \
		liblcms2-dev \
		libopenblas-dev \
		liblapack-dev \
		libpng12-dev \
		libssl-dev \
		libtiff5-dev \
		libwebp-dev \
		libzmq3-dev \
		nano \
		pkg-config \
		python-dev \
		software-properties-common \
		unzip \
		vim \
		wget \
		zlib1g-dev \
		qt5-default \
		libvtk6-dev \
		zlib1g-dev \
		libjpeg-dev \
		libwebp-dev \
		libpng-dev \
		libtiff5-dev \
		libjasper-dev \
		libopenexr-dev \
		libgdal-dev \
		libdc1394-22-dev \
		libavcodec-dev \
		libavformat-dev \
		libswscale-dev \
		libtheora-dev \
		libvorbis-dev \
		libopenjpeg5 \
		libxvidcore-dev \
		libx264-dev \
		yasm \
		libopencore-amrnb-dev \
		libopencore-amrwb-dev \
		libv4l-dev \
		libxine2-dev \
		libtbb-dev \
		libeigen3-dev \
		python-dev \
		python-tk \
		python-numpy \
		python3-dev \
		python3-tk \
		python3-numpy \
		ant \
		default-jdk \
		doxygen \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* && \
	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3
	# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)



# Install gibson-specific dependencies
RUN apt-get update && apt-get install -y \
		libglew-dev \
		libglm-dev \
		libassimp-dev \
		xorg-dev \
		libglu1-mesa-dev \
		libboost-dev \
		mesa-common-dev \
		freeglut3-dev \
		libopenmpi-dev \
		cmake \
		golang \
		libjpeg-turbo8-dev \
		wmctrl \ 
		xdotool \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/cache/apk/*

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# (Gibson) Install conda
#RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda2-4.3.21-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b && rm ~/miniconda.sh && \
#    export PATH=/home/ubuntu/miniconda2/bin:$PATH && \
#    echo "PATH=/home/ubuntu/miniconda2/bin:$PATH" >> ~/.bashrc &&

# Add SNI support to Python
RUN pip --no-cache-dir install \
		pyopenssl \
		ndg-httpsclient \
		pyasn1

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
		python-numpy \
		python-scipy \
		python-nose \
		python-h5py \
		python-skimage \
		python-matplotlib \
		python-pandas \
		python-sklearn \
		python-sympy \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install other useful Python packages using pip
RUN pip --no-cache-dir install --upgrade ipython && \
	pip --no-cache-dir install \
		Cython \
		ipykernel \
		jupyter \
		path.py \
		Pillow \
		pygments \
		six \
		sphinx \
		wheel \
		zmq \
		&& \
	python -m ipykernel.kernelspec

# Install other Gibson-specific Python packages using pip
RUN pip --no-cache-dir install --upgrade opencv-python cython

# (Gibson) install openai baselines
#RUN git clone https://github.com/openai/baselines.git && \
#	pip install -e baselines

# (Gibson) OpenGL support

# Install TensorFlow
#RUN pip --no-cache-dir install \
#	https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_ARCH}/tensorflow_${TENSORFLOW_ARCH}-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl

WORKDIR "/root"
#CMD ["/bin/bash"]
CMD ["sleep", "infinity"]
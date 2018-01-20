## Scratch ubuntu:16.04 image with NVIDIA GPU
FROM nvidia/cuda

## Skip keyboard settings
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install -y libav-tools \
    libpq-dev \
    libjpeg-dev \
    cmake \
    wget \
    unzip \
    git \
    xpra \
    golang-go \
    libboost-all-dev \
    make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \

## Install conda, opencv

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda2-4.3.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda install -c menpo opencv -y
#RUN conda install -y \
#    scikit-image \
#    flask \
#    pillow

RUN conda install pytorch torchvision cuda80 -c soumith

## Install Universe
WORKDIR /usr/local/realenv
RUN git clone https://github.com/openai/universe.git
WORKDIR /usr/local/realenv/universe
RUN pip install -e .

## Install Realenv
WORKDIR /usr/local/realenv
RUN wget https://www.dropbox.com/s/xmhgkmhgp9dfw52/realenv.tar.gz && tar -xvzf realenv.tar.gz && rm realenv.tar.gz
WORKDIR /usr/local/realenv/realenv
RUN pip install -e .

## Set up data & model for view synthesizer
#nvidia-docker run -it -v realenv-data:/usr/local/realenv/data realenv bash

## Start VNC server



## Entry point
WORKDIR /usr/local/realenv/
ENTRYPOINT [ "/bin/bash", "-c" ]


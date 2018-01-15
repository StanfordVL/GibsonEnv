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
    vnc4server \
    golang-go \
    libboost-all-dev \
    make \
    && apt-get clean
    ## This line raises error when building docker image
    # && rm -rf /var/lib/apt/lists/* \

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
RUN pip install progressbar

## Set up data & model for view synthesizer
#nvidia-docker run -it --rm -v realenv-data:/usr/local/realenv/data realenv bash

## Start VNC server

#RUN apt-get install -y x11vnc xvfb

#RUN mkdir ~/.vnc
# Setup a password
#RUN x11vnc -storepasswd 1234 ~/.vnc/passwd

COPY . /usr/local/realenv/

## Entry point
WORKDIR /usr/local/realenv/

RUN ["chmod", "+x", "/usr/local/realenv/init.sh"]


#ENTRYPOINT [ "/usr/local/realenv/init.sh" ]
#ENTRYPOINT [ "/bin/bash", "-c" ]
#CMD ["x11vnc", "-forever", "-usepw", "-create"]



LABEL io.k8s.description="Headless VNC Container with Xfce window manager, firefox and chromium" \
      io.k8s.display-name="Headless VNC Container based on Ubuntu" \
      io.openshift.expose-services="6901:http,5901:xvnc" \
      io.openshift.tags="vnc, ubuntu, xfce" \
      io.openshift.non-scalable=true

## Connection ports for controlling the UI:
# VNC port:5901
# noVNC webport, connect via http://IP:6901/?password=vncpassword
ENV DISPLAY :1
ENV VNC_PORT 5901
ENV NO_VNC_PORT 6901
EXPOSE $VNC_PORT $NO_VNC_PORT

ENV HOME /usr/local/realenv/
ENV STARTUPDIR /dockerstartup
WORKDIR $HOME

### Envrionment config
ENV DEBIAN_FRONTEND noninteractive
ENV NO_VNC_HOME $HOME/noVNC
ENV VNC_COL_DEPTH 24
ENV VNC_RESOLUTION 1280x1024
ENV VNC_PW vncpassword

RUN printf "qwertyui\nqwertyui\n\n" | vncpasswd

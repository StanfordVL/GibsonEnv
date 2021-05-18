GIBSON ENVIRONMENT for Embodied Active Agents with Real-World Perception
**************************************************************************
|BuildOnUbuntuLatest|_
|BuildManylinux20102014|_
|Gibson|_

The source code is available on this `Github repository`_.



*This package is generated starting from GibsonEnv project.
You can find the original source code* `here <https://github.com/StanfordVL/GibsonEnv>`_ *or you can visit the* `official website`_ .

**Summary**: Perception and being active (i.e. having a certain level of motion freedom) are closely tied. Learning active perception and sensorimotor control in the physical world is cumbersome as existing algorithms are too slow to efficiently learn in real-time and robots are fragile and costly. This has given a fruitful rise to learning in the simulation which consequently casts a question on transferring to real-world. We developed Gibson environment with the following primary characteristics:

**I.** being from the real-world and reflecting its semantic complexity through virtualizing real spaces,
**II.** having a baked-in mechanism for transferring to real-world (Goggles function), and
**III.** embodiment of the agent and making it subject to constraints of space and physics via integrating a physics engine `Bulletphysics`_.

**Naming**: Gibson environment is named after *James J. Gibson*, the author of "Ecological Approach to Visual Perception", 1979. “We must perceive in order to move, but we must also move in order to perceive” – JJ Gibson

Paper
=====

`Gibson Env: Real-World Perception for Embodied Agents <http://gibson.vision/>`_, in CVPR 2018 [Spotlight Oral].

Installation
=============

**CUDA Toolkit is necessary to run gibson!**

Installing precompiled version from pip
___________________________________________

Gibson can be simply installed from pip. The pip version of Gibson is precompiled only for linux machines. If you use another SO, you have to recompile Gibson from source.

.. code-block:: bash

    pip install gibson

Building from source
_______________________

If you don't want to use the precompiled version, you can also install gibson locally. This will require some dependencies to be installed.

First, make sure you have Nvidia driver and CUDA installed. If you install from source, CUDA 9 is not necessary, as that is for nvidia-docker 2.0.
Then, clone this repository recursively to download the submodules  and install the following dependencies:

.. code-block:: bash

    git clone https://github.com/micheleantonazzi/GibsonEnv.git --recursive
    apt-get update
    apt-get install doxygen libglew-dev xorg-dev libglu1-mesa-dev libboost-dev \
      mesa-common-dev freeglut3-dev libopenmpi-dev cmake golang libjpeg-turbo8-dev wmctrl \
      xdotool libzmq3-dev zlib1g-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev \
      libportmidi-dev libfreetype6-dev

Finally install the package using pip (during this process, Gibson is automatically compiled):

.. code-block:: bash

    pip install -e .


Install required deep learning libraries: Using python3 is recommended. You can create a python3 environment first.

Download Gibson assets
=======================

After the installation of Gibson, you have to set up the assets data (agent models, environments, etc).
The folder that stores the necessary data to run Gibson environment must be set by the user.
To do this, simply run this command `gibson-set-assets-path` in a terminal and then follow the printed instructions.
This script asks you to insert the path where to save the Gibson assets.
Inside this folder, you have to copy the environment core assets data
(available `here <https://storage.googleapis.com/gibson_scenes/assets_core_v2.tar.gz>`_ ~= 300MB)
and the environments data (downloadable from `here <https://storage.googleapis.com/gibson_scenes/dataset.tar.gz>`_ ~= 10GB).
The environment data must be located inside a sub-directory called `dataset`.
You can add more environments by adding them inside the `dataset` folder located in the previously set path.
Users can download and copy manually these data inside the correct path or they can use dedicated python utilities.
To easily download Gibson assets, typing in a terminal:

.. code-block:: bash

    gibson-set-assets-path # This command allows you to set the default Gibson assets folder
    gibson-download-assets-core
    gibson-download-dataset


.. |BuildManylinux20102014| image:: https://github.com/micheleantonazzi/GibsonEnv/actions/workflows/build_manylinux_2010_2014.yml/badge.svg?branch=master
.. |BuildOnUbuntuLatest| image:: https://github.com/micheleantonazzi/GibsonEnv/actions/workflows/build_ubuntu_latest.yml/badge.svg?branch=master
.. |Gibson| image:: https://img.shields.io/pypi/v/gibson.svg
.. _BuildManylinux20102014: https://github.com/micheleantonazzi/GibsonEnv/actions/workflows/build_manylinux_2010_2014.yml
.. _BuildOnUbuntuLatest: https://github.com/micheleantonazzi/GibsonEnv/actions/workflows/build_ubuntu_latest.yml/badge.svg
.. _Gibson: https://pypi.org/project/gibson
.. _Github repository: https://github.com/micheleantonazzi/GibsonEnv
.. _official website: http://gibsonenv.stanford.edu/
.. _Bulletphysics: http://bulletphysics.org/wordpress/

GIBSON ENVIRONMENT for Embodied Active Agents with Real-World Perception
**************************************************************************

|ImageLink|_

`Github repository`_

`Website`_

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

C. Building from source
_______________________

If you don't want to use the precompiled version, you can also install gibson locally. This will require some dependencies to be installed.

First, make sure you have Nvidia driver and CUDA installed. If you install from source, CUDA 9 is not necessary, as that is for nvidia-docker 2.0. Then, let's install some dependencies:

.. code-block:: bash

    apt-get update
    apt-get install doxygen libglew-dev xorg-dev libglu1-mesa-dev libboost-dev \
      mesa-common-dev freeglut3-dev libopenmpi-dev cmake golang libjpeg-turbo8-dev wmctrl \
      xdotool libzmq3-dev zlib1g-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev \
      libportmidi-dev libfreetype6-dev

Then, cloning the repository and install the package using pip
.. code-block:: bash

    pip install -e .


Install required deep learning libraries: Using python3 is recommended. You can create a python3 environment first.

.. |ImageLink| image:: https://github.com/micheleantonazzi/GibsonEnv/actions/workflows/build_manylinux_2010.yml/badge.svg?branch=pip-build
.. _ImageLink: https://github.com/micheleantonazzi/GibsonEnv/actions/workflows/build_manylinux_2010.yml
.. _Github repository: https://github.com/StanfordVL/GibsonEnv
.. _Website: http://gibsonenv.stanford.edu/
.. _Bulletphysics: http://bulletphysics.org/wordpress/

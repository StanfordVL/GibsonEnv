GIBSON ENVIRONMENT for Embodied Active Agents with Real-World Perception
**************************************************************************

|ImageLink|_

`Github_repository`_

`Website`_

**Summary**: Perception and being active (i.e. having a certain level of motion freedom) are closely tied. Learning active perception and sensorimotor control in the physical world is cumbersome as existing algorithms are too slow to efficiently learn in real-time and robots are fragile and costly. This has given a fruitful rise to learning in the simulation which consequently casts a question on transferring to real-world. We developed Gibson environment with the following primary characteristics:

**I.** being from the real-world and reflecting its semantic complexity through virtualizing real spaces,
**II.** having a baked-in mechanism for transferring to real-world (Goggles function), and
**III.** embodiment of the agent and making it subject to constraints of space and physics via integrating a physics engine `Bulletphysics`_.

**Naming**: Gibson environment is named after *James J. Gibson*, the author of "Ecological Approach to Visual Perception", 1979. “We must perceive in order to move, but we must also move in order to perceive” – JJ Gibson

#### Paper
**["Gibson Env: Real-World Perception for Embodied Agents"](http://gibson.vision/)**, in **CVPR 2018 [Spotlight Oral]**.


[![Gibson summary video](misc/vid_thumbnail_600.png)](https://youtu.be/KdxuZjemyjc "Click to watch the video summarizing Gibson environment!")
.. |ImageLink| image:: https://github.com/micheleantonazzi/GibsonEnv/actions/workflows/build_manylinux.yml/badge.svg?branch=pip-build
.. _ImageLink: https://github.com/micheleantonazzi/GibsonEnv/actions/workflows/build_manylinux.yml
.. _Github_repository: https://github.com/StanfordVL/GibsonEnv
.. _Website: http://gibsonenv.stanford.edu/
.. _Bulletphysics: http://bulletphysics.org/wordpress/

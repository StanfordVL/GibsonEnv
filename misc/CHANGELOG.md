# Changelog
Notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).
https://github.com/StanfordVL/GibsonEnv/blob/master/README.md
## 0.5.0 - 2021-05-03
### Added
 - **Improved the build procedure and published the compiled version on pypi (for now test pypi)**
 - Improve build procedure: now it is performed in setup.py
 - Added a github action which builds the project (using Python 2.7, 3.6, 3.7, 3.8, 3.9) and publishes the compiled version on pypi
 - Changed the assets default path: now it must be chosen by the user using `gibson-set-assets-path` command
 - Deleted the assets data folder: all files are contained in assets-core-v2.tar.gz
 - Created AssetsManager: it stores the assets path and other configuration parameters
### Issues

The docker image must and the related procedure to install Gibson must be fixed.

## 0.4.0 - 2021-04-29
### Added
 - Build procedure tested in Ubuntu 20.04 LTS
 - Fixed build issues
 - Fixed dependencies issues
 - Fixed CMake issues
 - Added new environment: TurtlebotNavigateNoPhysicsEnv. In this environment, all physical constraints are deleted. The gravity is set to zero and the collisions between the agent and the environment are ignored. Therefore, the only way to move the agent is to manually set its position at each stage of the simulation.
## 0.3.1 - 2018-08-11
### Added
 - EGL integration, remove X server dependency (solve #16 #24 #25)
 - OpenAI Gym 0.10.5 compatibility
 - Updated rendering filler models, added unfiller models
 - Bug fixes

## 0.3.0 - 2018-06-18
### Added
 - Full dataset
 - ROS integration
 - Misc bug fixes

## 0.2.1 - 2018-04-18
Bug fixes
### Fixed
- Bug reported by @jackbruce. Random initialization for robot initial position.



## 0.2.0 - 2018-03-13
MINOR adds and PATCH fixes. [Commit](https://github.com/StanfordVL/GibsonEnv/commit/69ae7ea348d1af9821bdc7ed124f1e46fc9e6479)
### Added
- Environment offers `self.robot` API for state, observation, orientation, action, eye, reset, etc. 
- You can define your own environment with customized rewards. See Semantic [README](https://github.com/StanfordVL/GibsonEnv/blob/master/README.md) for instructions.

### Changed
- `env.step(action)` now returns as first value `obs` a dictionary. You can get each component by `obs['rgb_filled'], obs['depth']`.
- RL example code modified to reflect the new return value of `env.step(action)`.
- `is_discrete` is specified inside configuration file, instead of env class initialization. Keeping it clean.

### Fixed
- Intricacy in environment class inheritance. Now parent class does not assume child class attributes. 
- Issue with rendering `rgb_prefilled`, `rgb_filled`. 
- Resolved issue of agent being blocked by a 'transparent wall'.
- Removed unnecessary logging.

## 0.1.0 - 2018-02-26
Initial beta release.

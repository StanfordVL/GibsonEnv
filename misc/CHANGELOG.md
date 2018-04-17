# Changelog
Notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).
https://github.com/StanfordVL/GibsonEnv/blob/master/README.md
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

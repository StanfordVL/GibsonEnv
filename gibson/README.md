## Gibson Core Environment

This folder contains the code base for running universe engine, as well as instructions needed for implementing custom environment and agent
### File Structure
 - Client: client side code for running remote environment
 - Core: realenv engine
 - Data: realenv dataset
 - Envs: repository of current environments, each can be thought of as a "challenge"
 - Spaces: Envs dependencies

### Implementing Agent

```python
observation, reward, done, info = env._step({})
```

 - *observation* (object): agent's observation of the current environment
 - *reward* (float) : amount of reward returned after previous action
 - *done* (boolean): whether the episode has ended. The agent is responsible for taking care of this, by calling `env.restart()`
 - *info* (dict): auxiliary diagnostic information (helpful for debugging, and sometimes learning)


Instruction for Using Gibson with Semantics
==========================

Gibson can provide semantics from:
0. Random semantics
...Good for visualization purpose
...in config.yaml set semantic_source: 0
1. Stanford 2D3Ds
...in config.yaml set semantic_source: 1
2. Matterport 3D
...in config.yaml set semantic_source: 2

## Instruction
1. Acquire data

Download [Stanford 2D3Ds dataset](https://github.com/alexsax/2D-3D-Semantics) or [Matterport 3D dataset](https://github.com/niessner/Matterport). 

2. Set environment variables
```bash
GIBSON_SEMANTIC_DSET=/path/to/dataset
```
3. Execute dataascript
```bash
python gibson/utils/semantic_data.py
```

Then, enjoy! Sample code:
```bash
python gibson/utils/test_env.py
```
Note: semantic mesh model might take up to a minute to load and parse. If you observe the loading
script taking longer than normal, please start an issue.

## Data Format
Default color coding is defined inside `gibson/core/channels/common/semantic_color.hpp`
For both 2D3Ds and Matterport3D, the rgb color is encoded as:
``` cpp
b = ( segment_id ) % 256;
g = ( segment_id >> 8 ) % 256;
r = ( segment_id >> 16 ) % 256;
color = {r, g, b}
```
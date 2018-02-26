Instruction for Using Gibson with Semantics
==========================

<img src=misc/semantics.png width="600">

Gibson can provide pixel-wise frame-by-frame semantic masks when the model is semantically annotated. As of now we have incorporated models from [Stanford 2D3DS](http://buildingparser.stanford.edu/) and [Matterport 3D](https://niessner.github.io/Matterport/) for this purpose, and we refer you to the original dataset's reference for the list of their semantic classes and annotations. 

## Dataset
Gibson can provide semantics from:

0. Random semantics <br />
Assigns a arbitrary distinctive color to each object. Good for visualization purpose <br />
In `config.yaml` set `semantic_source: 0`
1. Stanford 2D3Ds <br />
In `config.yaml` set `semantic_source: 1`
2. Matterport 3D <br />
In `config.yaml` set `semantic_source: 2`

## Instruction
1. Acquire data<br />
**Agreement**: If you choose to use the models from [Stanford 2D3DS](http://buildingparser.stanford.edu/) or [Matterport 3D](https://niessner.github.io/Matterport/)for rendering semantics, we ask you to agree to and sign their respective agreements. See [here](https://niessner.github.io/Matterport/) for Matterport3D and [here](https://github.com/alexsax/2D-3D-Semantics) for Stanford 2D3DS.

2. Move model files to gibson/assets/dataset<br />
Then, enjoy! Sample code:
```bash
python examples/demo/play_husky_semantics.py
```
Note: semantic mesh model might take up to a minute to load and parse. If you observe the script takes longer than normal, likely it's because of OpenGL rendering issue. Please file an issue and we will help you with troubleshooting.

## Data Format
Default color coding is defined inside `gibson/core/channels/common/semantic_color.hpp`
For both 2D3Ds and Matterport3D, the rgb color is encoded as:
``` cpp
b = ( segment_id ) % 256;
g = ( segment_id >> 8 ) % 256;
r = ( segment_id >> 16 ) % 256;
color = {r, g, b}
```
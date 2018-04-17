Instruction for using Gibson with Semantics
==========================

## Dataset Agreement
Gibson has incorporated models from [Stanford 2D3DS](http://buildingparser.stanford.edu/) and [Matterport 3D](https://niessner.github.io/Matterport/). If you choose to use these models for rendering semantics, we ask you to agree to and sign their respective agreements. See [here](https://niessner.github.io/Matterport/) for Matterport3D and [here](https://github.com/alexsax/2D-3D-Semantics) for Stanford 2D3DS.

In the light beta release, the space `17DRP5sb8fy` includes Matterport 3D style semantic annotation and `space7` includes Stanford 2D3DS style annotation. 

## Quickstart

Inside config.yaml, fill in the following fields to render semantics (see `examples/configs/husky_navigate_semantics.yaml for sample use case`).

```yaml
...
use_filler: true
display_ui: true
show_diagnostics: true
ui_num: 2                                       # Make sure match up with len(ui_components)
ui_components: [RGB_FILLED, SEMANTICS]          # Make sure to include SEMANTICS
output: [nonviz_sensor, rgb_filled, semantics]  # Make sure to include semantics
...
mode: gui
semantic_source: 1                              # 1 for Stanford 2D3Ds, 2 for MP3D 
semantic_color: 1                               # 1 for distinctive color, 2 for label index rgb code
```

Sample code:
```bash
python examples/demo/play_husky_semantics.py
```
Note: semantic mesh model might take up to a minute to load and parse. If you observe the script takes longer than normal, likely it's because of OpenGL rendering issue. Please file an issue and we will help you with troubleshooting.

Configuration Argument

| Argument name        | Example value           | Explanation  |
|:-------------:|:-------------:| :-----|
| semantic_source      | 1 | Using Stanford 2D3Ds for semantic source |
| semantic_source      | 2 | Using Matterport3D for semantic source |
| semantic_color       | 1 | instance-by-instance color coding scheme | 
| semantic_color       | 2 | Semantic label color coding scheme |

## Semantic Color Coding
There are two ways for rendering rgb semantic maps in semantic mode, defined inside `gibson/core/channels/common/semantic_color.hpp`. Each is defined below:

###  Instance-by-Instance Color Coding

<img src=../../misc/instance_colorcoding_semantics.png width="600">

In Instance-by-Instance Color Coding Scheme, the environment assigns a random distinctive color to each semantic instance. These colors are arbitrarily chosen but are preserved through different trials. Note that this mode renders intuitive colorful semantic map frames, but the rgb values do not enable easy semantic class lookup.


###  Semantic Label Color Coding

<img src=../../misc/semanticlabels_colorcoding_semantics.png width="600">

In Semantic Label Color Coding, for both Stanford 2D3Ds and Matterport3D spaces, the environment assigns a semantic label to each object instance and renders the frame. These integer labels usually have their space specific mapping, specified by the original Stanford 2D3Ds and Matterport3D datasets. Instead of enforcing another layer of indirection, Gibson directly renders these semantic labels as rgb; therefore, the rendered frame can be directly consumed as the semantically labled pixel maps but the frame itself usually looks dark to human eyes. 

``` cpp
b = ( segment_id ) % 256;
g = ( segment_id >> 8 ) % 256;
r = ( segment_id >> 16 ) % 256;
color = {r, g, b}
```

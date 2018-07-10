# Full Gibson Environment Dataset

Full Gibson Environment Dataset consists of 572 models and 1440 floors. We cover a diverse set of models including households, offices, hotels, venues, museums, hospitals, construction sites etc. 
<img src=../../misc/spaces.png width="800">

Table of contents
=================

   * [Download](#download)
   * [Dataset Modalities](#dataset-modalities)
   * [Dataset Splits](#dataset-splits)
   * [Navigation Benchmark Scenarios](#navigation-benchmark-scenarios)
   * [Dataset Metrics](#dataset-metrics)

## Download
The link will first take you to a license agreement, and then to the data.

### [[ Download the full Gibson Dataset ]](http://gibson.vision)  [[ checksums ]](gibson.vision)

#### License Note: The dataset license is included in the above link. The license in this repository covers only the provided software.

### Citations
If you use this dataset please cite:
```
@inproceedings{xiazamirhe2018gibsonenv,
  title={Gibson env: real-world perception for embodied agents},
  author={Xia, Fei and R. Zamir, Amir and He, Zhiyang and Sax, Alexander and Malik, Jitendra and Savarese, Silvio},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2018 IEEE Conference on},
  year={2018},
  organization={IEEE}
}
```


## Dataset Modalities
Each model in the dataset has its own folder in the dataset. All the modalities and metadata for each area are contained in that folder. 
```
/pano
  /points		 # camera metadata
  /rgb			 # rgb images
  /mist			 # depth images
  /normal		 # surface normal images
mesh.obj 		 # 3d mesh
mesh_z_up.mtl 	 # 3d mesh for physics engine
camera_poses.csv # camera locations
semantic.obj (optional) # 3d mesh with semantic annotation
```

## Dataset Splits
We provide three different standard splits for our dataset. 

| Split Name   |      Train     |  Val  |  Test | Hole Filled |
|----------|:-------------:|-------------:|------:| ------:|
| Tiny |  25 | 5 | 5 | 100% |
| Medium |  100 |  20 | 20 | 100% |
| Full | 360 | 70 | 70 | 100% |
| Full+ | 412 |  80 | 80 | 90.9% |

**Hole Filling**: We applied combination of automatic and manual hole-filling techniques on `tiny`, `medium` and `full` sets, to ensure that the models do not have severe reconstruction artifacts. `full+` contains the rest of the models that we are incapable of hole-filling, based on current techniques.  
**Split Criteria**: In every split, we sort all 572 models by the linear combination of `loor number`, `area`, `ssa` and `navigation complexity`. We select a `tiny` as the set of models with the highest combination scores. We also set `medium` to be inclusive of `tiny`, and `full` to be inclusive of `mediuim`.

# Navigation Benchmark Scenarios

We define navigation scenarios in [Gibson Standard Navigation Benchmark](https://www.dropbox.com/sh/9wpego1rswwbbm8/AAD_e6ZwXU4tzniaBidkdXIwa?dl=0). The following column values are provided for each episode:

- `split`: `train`, `val`, or `test` indicating split for the episode.
- `task`: string id for task type, currently restricted to `p` for `point_goal`.
- `sceneId`: id of model within which episode takes place.
- `level`: integer id of level (typically floor) within scene, starting from `0`.
- `startX`, `startY`, `startZ`: coordinates of agent starting position in scene space.
- `startAngle`, : azimuth angle (counter-clockwise from scene space +X axis) of agent starting state.
- `goalRoomId`, `goalRoomType`: currently not available.
- `goalObjectId`, `goalObjectType`: currently not available.
- `goalX`, `goalY`, `goalZ`: coordinates of goal point in scene space. Required for all task types. Position of goal for `point_goal`.
- `dist`, `pathDist`: Euclidean and geodesic (along shortest path) distance from agent start position to goal position
- `pathNumDoors`, `pathDoorIds`: currently not available.
- `pathNumRooms`, `pathRoomIndices`: currently not available.


## Dataset Metrics

**Floor Number** Total number of floors in each model.

We calculate floor numbers by using camera sweeping locations. We use `sklearn.cluster.DBSCAN` to cluster these locations by height and set minimum cluster size to `5`. This means areas with at least `5` sweeps are treated as one single floor. This helps us capture small building spaces such as backyard, attics, basements.

**Area** Total floor area of each model.

We calculate total floor area by summing up area of each floor. This is done by sampling point cloud locations based on floor height, and fitting a `scipy.spatial.ConvexHull` on sample locations.

**SSA** Specific surface area. 

The ratio of inner mesh surface and volume of convex hull of the mesh. This is a measure of clutter in the models: if the inner space is placed with large number of furnitures, objects, etc, the model will have high SSA. 

**Navigation Complexity** The highest complexity of navigating between arbitrary points within the model.

We sample arbitrary point pairs inside the model, and calculate `A∗` navigation distance between them. `Navigation Complexity` is equal to `A*` distance divide by `straight line distance` between the two points. We compute the highest navigation complexity for every model. Note that all point pairs are sample within the *same floor*.

**Subjective Attributes**

We examine each model manually, and note the subjective attributes of them. This includes their furnishing style, house shapes, whether they have long stairs, etc.
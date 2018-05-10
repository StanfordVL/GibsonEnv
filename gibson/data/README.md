# Full Gibson Environment Dataset

Full Gibson Environment Dataset consists of 572 models and 1440 floors. We cover a diverse set of models including households, offices, hotels, venues, museums, hospitals, construction sites etc. You can contact. We have included [spec sheet](https://docs.google.com/spreadsheets/d/1hhjAtgASv8MBkXa7aXH5obyf7v6d-oDm1KRR0oiSEzk/edit?usp=sharing) of the full Gibson Dataset where you can look up individual models and their information

<img src=misc/spaces.png width="800">

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

We examine each model manually, and note the subjective attributes of them. This includes their furnishing style, room shapes, whether they have long stairs, etc.

## Train/Test Set
We divide up 572 models into 515 training models (90%) and 57 testing models (10%). We sort all 572 models by the linear combination of `floor number`, `area`, `ssa` and `navigation complexity`. We select a diverse set of models as test set by selecting the 10% models with highest combination scores. 




# Data Generation Script for Real Env

### Requirements
Recommend: create virtual environment for the dependencies
Install Blender 2.79
```shell
sudo add-apt-repository ppa:thomas-schiex/blender
sudo apt-get update
sudo apt-get install blender
```
Create necessary environment
```shell
conda create -n (env_name) python=3.5
pip install pillow
curl -sL https://deb.nodesource.com/setup_8.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install optimist bytebuffer long
```


### Run the code
```shell
source activate (env_name)
python start.py
```

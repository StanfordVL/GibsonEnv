## Usage: random_husky.py debugging
import os
from gibson.data.datasets import get_model_path


def run_depth_render():
    model_path = get_model_path()[0]
    dr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'depth_render')
    os.chdir(dr_path)
    print("./depth_render --modelpath {}".format(model_path))
    os.system("./depth_render --modelpath {}".format(model_path))
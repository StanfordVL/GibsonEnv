import os, yaml
import numpy as np

MODEL_ROOT = "/mnt/sdc/Gibson_Models/mp3d/"
CONFIG_PATH = "/home/jerry/Desktop/gibson-test/GibsonEnv/examples/configs/recording"
SAMPLE_CONFIG = "/home/jerry/Desktop/gibson-test/GibsonEnv/examples/configs/record_.yaml"


template = {}

with open(SAMPLE_CONFIG, 'r') as stream:
    try:
        template = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

for file in os.listdir(CONFIG_PATH):
    os.remove(os.path.join(CONFIG_PATH, file))

for model_id in os.listdir(MODEL_ROOT):
    template["envname"] = "DroneNavigateEnv"
    
    template["ui_num"] = 3
    template["ui_components"] = ["RGB_FILLED", "DEPTH", "SEMANTICS"]
    template["output"] = ["rgb_filled", "depth", "semantics"]
    #template["ui_num"] = 2
    #template["ui_components"] = ["RGB_FILLED", "DEPTH"]
    #template["output"] = ["rgb_filled", "depth"]

    template["semantic_source"] = 2 # Matterport 3D
    template["semantic_color"] = 1  # Random
    #template['model_id'] = "stadium"
    template['model_id'] = model_id
    template['is_discrete'] = False
    template['speed']['frameskip'] = 20
    template['speed']['timestep'] = 0.001
    template['fov'] = 2.094
    #template['display_ui'] = False
    template["root"] = "/media/Drive3/Gibson_Models/mp3d"

    if not os.path.isdir(os.path.join(MODEL_ROOT, model_id)): continue
    if not os.path.isfile(os.path.join(MODEL_ROOT, model_id, 'pair.csv')): continue
    f_csv = open(os.path.join(MODEL_ROOT, model_id, 'pair.csv'))
    for line in f_csv:
        p_index = int(line.split(",")[0])
        p_pos_start = [float(v.strip()) for v in line.split(",")[1:4]]
        p_pos_end = [float(v.strip()) for v in line.split(",")[4:7]]
        p_orn = [float(v.strip()) for v in line.split(",")[7:]]
        #p_pos[2] -= 0.5
        p_orn[2] -= np.pi/2
        template['initial_pos'] = p_pos_start
        template['initial_orn'] = p_orn
        template['target_pos'] = p_pos_end
        template['point_num'] = p_index
        fout = open(os.path.join(CONFIG_PATH, "record_{}_{}.yaml".format(model_id, p_index)), "w+")
        yaml.dump(template, fout, default_flow_style=False)
        fout.close()
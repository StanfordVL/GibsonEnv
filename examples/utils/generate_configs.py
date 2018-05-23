import os, yaml
import numpy as np

MODEL_ROOT = "/mnt/sdc/Gibson_Models/572_processed/"
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
    template['model_id'] = model_id
    if not os.path.isdir(os.path.join(MODEL_ROOT, model_id)): continue
    if not os.path.isfile(os.path.join(MODEL_ROOT, model_id, 'sample.csv')): continue
    f_csv = open(os.path.join(MODEL_ROOT, model_id, 'sample.csv'))
    for line in f_csv:
        p_index = int(line.split(",")[0])
        p_pos = [float(v.strip()) for v in line.split(",")[1:4]]
        p_orn = [float(v.strip()) for v in line.split(",")[4:]]
        p_pos[2] -= 0.5
        p_orn[2] -= np.pi / 4
        template['initial_pos'] = p_pos
        template['initial_orn'] = p_orn
        template['point_num'] = p_index
        fout = open(os.path.join(CONFIG_PATH, "record_{}_{}.yaml".format(model_id, p_index)), "w+")
        yaml.dump(template, fout, default_flow_style=False)
        fout.close()
import os
from subprocess import call, Popen, PIPE
from threading import Timer
import sys

full_csv_name = "split_full.csv"
full_models = []
medium_csv_name = "split_medium.csv"
medium_models = []
tiny_csv_name = "split_tiny.csv"
tiny_models = []

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), full_csv_name), 'r') as f:
    for line in f:
        full_models.append(line.strip())
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), medium_csv_name), 'r') as f:
    for line in f:
        medium_models.append(line.strip())
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), tiny_csv_name), 'r') as f:
    for line in f:
        tiny_models.append(line.strip())

RECORD_ROOT = "/media/Drive3/Gibson_Models/572_avi"

all_models = []

for m in tiny_models + medium_models + full_models:
    if m not in all_models: all_models.append(m)


for m in sorted(all_models):
    if os.path.isdir(os.path.join(RECORD_ROOT, m)) and len(os.listdir(os.path.join(RECORD_ROOT, m))) > 0: continue
    cmd_line = "/home/jerry/Desktop/gibson-test/GibsonEnv/examples/demo/record_drone.py --model_id {}".format(m)
    proc = Popen([sys.executable] + cmd_line.split())    
    timer = Timer(150, proc.kill)
    try:
        timer.start()
        proc.wait()
        print("Finished: %s" % m)
    finally:
        timer.cancel()
    os.system("pkill depth")
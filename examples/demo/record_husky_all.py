import os
from subprocess import call, Popen, PIPE
from threading import Timer
import sys

csv_name = "split_full.csv"
models = []

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_name), 'r') as f:
    for line in f:
        models.append(line.strip())

for m in models[4:]:
    cmd_line = "/home/jerry/Desktop/gibson-test/GibsonEnv/examples/demo/record_husky.py --model_id {}".format(m)
    proc = Popen([sys.executable] + cmd_line.split())    
    timer = Timer(300, proc.kill)
    try:
        timer.start()
        proc.wait()
        print("Finished: %s" % m)
    finally:
        timer.cancel()
    os.system("pkill depth")
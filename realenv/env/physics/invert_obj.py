import os
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--datapath'  , required = True, help='dataset path')
	parser.add_argument('--model'  , type = str, default = '', help='path of model')

	opt = parser.parse_args()
	
	input_file  = os.path.join(opt.datapath, opt.model, 'modeldata', 'out_res.obj')
	output_file = os.path.join(opt.datapath, opt.model, 'modeldata', 'out_z_up.obj')

	f_original = open(input_file)
	f_inverted = open(output_file, 'w+')

	for line in f_original:
		if line[:2] == 'v ':
			line = line.split()
			inverted = ['v', str(float(line[1])), str(-float(line[3])), str(float(line[2])), '\n']
			inverted = " ".join(inverted)
			f_inverted.write(inverted)
		else:
			f_inverted.write(line)

	f_original.close()
	f_inverted.close()
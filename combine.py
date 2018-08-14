import json

data = {}

with open("fps_128.json", "r") as f:
	data = json.load(f)

with open("fps_256.json", "r") as f:
	new_data = json.load(f)
	for m in new_data.keys():
		for r in new_data[m].keys():
			data[m][r] = new_data[m][r] 

with open("fps_512.json", "r") as f:
	for m in new_data.keys():
		for r in new_data[m].keys():
			data[m][r] = new_data[m][r] 

with open("fps.json", "w") as f:
	json.dump(data, f, indent=4)
# Reformat texture file names
# Incorrect: 73e85207dc844aa4b814c73120c3b53f_texture_jpg_high%2F73e85207dc844aa4b814c73120c3b53f_005.jpg
# Correct  : 73e85207dc844aa4b814c73120c3b53f_005.jpg

import os
for filename in os.listdir("."):
	if not '_50k' in filename and ".jpg" in filename:
		start = filename.index('_texture')
		end   = len(filename) - filename[::-1].index('_') - 1
		os.rename(filename, filename[:start] + filename[end:])
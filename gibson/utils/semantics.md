Instruction for Using Gibson with Semantics

Gibson can provide semantics from:
0. Random semantics
	Good for visualization purpose
	in .yaml set source:0
1. Stanford 2D3Ds
	in .yaml set source:1
2. Matterport 3D
	in .yaml set source:2

Instruction
1. Acquire data
	you need to email xxx
2. Set environment variables

3. Execute data script


Then, enjoy!
Note: semantic mesh model might take up to a minute to load and parse. If you observe the loading
script taking longer than normal, please start an issue.
Sample code:
	python gibson/utils/test_env.py


Data Format
Default color coding is defined inside gibson/core/channels/common/semantic_color.hpp
For both 2D3Ds and Matterport3D, the rgb color is encoded as:
    b = ( segment_id ) % 256;
    g = ( segment_id >> 8 ) % 256;
    r = ( segment_id >> 16 ) % 256;
    color = {r, g, b}
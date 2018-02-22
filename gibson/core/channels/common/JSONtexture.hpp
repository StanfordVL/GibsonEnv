#include "picojson.h"
#include "MTLtexture.hpp"

#include <sstream>
/* main function to parse MTL files, load or generate texture iamges and generate openGL texture IDs */
bool loadJSONtextures (std::string jsonpath, std::vector<TextureObj> & objText, std::vector<std::string> OBJmaterial_name) {
    std::string imagePath;

    std::string fsegs_json = jsonpath + "out_res.fsegs.json";
		std::ifstream ss(fsegs_json, std::ios::binary);
    if (ss.fail())
        throw std::runtime_error("failed to open " + fsegs_json);

    std::vector<int> segid_list;
    std::vector<std::vector<int>> segid_to_index;
    std::vector<int> index_to_group_id;
    std::vector<int> index_to_group_label;

    int max_segid = -1;

	  if (ss) {
	    // get length of file:
	    ss.seekg (0, ss.end);
	    int length = ss.tellg();
	    ss.seekg (0, ss.beg);

	    char * buffer = new char [length];
	    std::cout << "Reading " << length << " characters... ";
	    ss.read (buffer,length);
	    if (ss)
	      std::cout << "all characters read successfully." << std::endl;
	    else
	      std::cout << "error: only " << ss.gcount() << " could be read" << std::endl;
	    ss.close();

	    // ...buffer contains the entire file...
	    picojson::value v;
			std::string err;
			const char* json_end = picojson::parse(v, buffer, buffer + strlen(buffer), &err);
			if (! err.empty())
			  std::cerr << err << std::endl;

			// check if the type of the value is "object"
			if (! v.is<picojson::object>()) {
			  std::cerr << "JSON is not an object" << std::endl;
			  exit(2);
			}

			// obtain a const reference to the map, and print the contents
			picojson::array value_arr;
			const picojson::value::object& obj = v.get<picojson::object>();
			for (picojson::value::object::const_iterator i = obj.begin();
			     i != obj.end();
			     ++i) {
				if (i-> first == "segIndices" && i->second.is<picojson::array>()) {
					value_arr = i->second.get<picojson::array>();
				}
			}

			for (unsigned int i = 0; i < value_arr.size(); i++) {
				int val = (stoi(value_arr[i].to_str()));
				segid_list.push_back(val);
				index_to_group_id.push_back(0);
				index_to_group_label.push_back(0);
				if (val > max_segid)
					max_segid = val;
			}

			for (unsigned int i = 0; i < max_segid + 1; i++)
				segid_to_index.push_back(std::vector<int>());
			for (unsigned int i = 0; i < segid_list.size(); i++) {
				segid_to_index[segid_list[i]].push_back(i);
			}

	    delete[] buffer;
	  }

	  printf("Finished loading indexes: %d\n", segid_list.size());

		std::string ssegs_json = jsonpath + "out_res.semseg.json";
		std::ifstream sg(ssegs_json, std::ios::binary);
    if (sg.fail())
        throw std::runtime_error("failed to open " + ssegs_json);

    std::vector<std::vector<int>> seg_groups;
	  if (sg) {
	    // get length of file:
	    sg.seekg (0, sg.end);
	    int length = sg.tellg();
	    sg.seekg (0, sg.beg);

	    char * buffer = new char [length];
	    std::cout << "Reading " << length << " characters... ";
	    sg.read (buffer,length);
	    if (sg)
	      std::cout << "all characters read successfully." << std::endl;
	    else
	      std::cout << "error: only " << sg.gcount() << " could be read" << std::endl;
	    sg.close();

	    // ...buffer contains the entire file...
	    picojson::value v;
			std::string err;
			const char* json_end = picojson::parse(v, buffer, buffer + strlen(buffer), &err);
			if (! err.empty())
			  std::cerr << err << std::endl;

			// check if the type of the value is "object"
			if (! v.is<picojson::object>()) {
			  std::cerr << "JSON is not an object" << std::endl;
			  exit(2);
			}

			// obtain a const reference to the map, and print the contents
			picojson::array segment_arr;
			const picojson::value::object& obj = v.get<picojson::object>();
			for (picojson::value::object::const_iterator i = obj.begin();
			     i != obj.end();
			     ++i) {
				if (i-> first == "segGroups" && i->second.is<picojson::array>()) {
					segment_arr = i->second.get<picojson::array>();
				}
			}

			picojson::array::iterator it;
      for (it = segment_arr.begin(); it != segment_arr.end(); it++) {
      	picojson::object obj_it = it->get<picojson::object>();
      	int group_id = stoi(obj_it["id"].to_str());
      	int group_label = stoi(obj_it["label_index"].to_str());
      	
      	picojson::value seg_val = obj_it["segments"];
      	picojson::array seg_arr_i;
      	if (seg_val.is<picojson::array>()) {
					seg_arr_i = seg_val.get<picojson::array>();
					for (unsigned int i = 0; i < seg_arr_i.size(); i++) {
						int segid = (stoi(seg_arr_i[i].to_str()));
						//if (group_id == 171)
							//std::cout << "\t sub index " << i << " " << index << " size " << seg_arr_i.size() << " group id size " << index_to_group_id.size() << " group label size " << index_to_group_label.size() << std::endl;
						//if (group_id == 1) {
						//	std::cout << "\t sub index " << i << "/" << seg_arr_i.size() << " segid " << segid << " original index " << segid_to_index[segid] << std::endl;
						//}
						for (unsigned int j = 0; j < segid_to_index[segid].size(); j++) {
							int index = segid_to_index[segid][j];
							index_to_group_id[index] 		= group_id;
		    			index_to_group_label[index] = group_label;
						}
		    	}
				}
      }

	    delete[] buffer;
	  }

	  //std::cout << "index to group id " << index_to_group_id[0] << " label " << index_to_group_label[0] << std::endl;
	  //std::cout << "index to group id " << index_to_group_id[1] << " label " << index_to_group_label[1] << std::endl;
	  //std::cout << "index to group id " << index_to_group_id[6] << " label " << index_to_group_label[6] << std::endl;
    return true;
}

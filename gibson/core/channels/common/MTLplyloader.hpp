#ifndef MTLPLYLOADER_H
#define MTLPLYLOADER_H

#include "MTLtexture.hpp"


bool loadPLY_MTL(
    const char * path,
    std::vector<std::vector<glm::vec3>> & out_vertices,
    std::vector<std::vector<glm::vec2>> & out_uvs,
    std::vector<std::vector<glm::vec3>> & out_normals,
    std::vector<std::string> & out_material_name,
    std::string & out_mtllib,
    int & num_vertices
);

bool loadJSONtextures (
	std::string jsonpath, 
	std::vector<std::string> & textureNames, 
	std::vector<int> textureIDs
);

#endif

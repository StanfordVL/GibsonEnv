#ifndef MTLPLYLOADER_H
#define MTLPLYLOADER_H

#include "MTLtexture.hpp"


bool loadPLY_MTL(
    const char * path,
    std::vector<std::vector<glm::vec3>> & out_vertices,
    std::vector<std::vector<glm::vec2>> & out_uvs,
    std::vector<std::vector<glm::vec3>> & out_normals,
    std::vector<glm::vec3> & out_centers,
    //std::vector<int> & out_material_name,
    std::vector<int> & out_material_id,
    std::string & out_mtllib,
    int & num_vertices
);

bool loadJSONtextures (
    std::string jsonpath, 
    std::vector<int> & out_label_id, 
    std::vector<int> & out_segment_id,
    std::vector<std::vector<int>> & out_face_indices
);

#endif

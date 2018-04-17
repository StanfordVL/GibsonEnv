#ifndef MTL_OBJLOADER_H
#define MTL_OBJLOADER_H

bool loadOBJ_MTL(
    const char * path,
    std::vector<std::vector<glm::vec3>> & out_vertices,
    std::vector<std::vector<glm::vec2>> & out_uvs,
    std::vector<std::vector<glm::vec3>> & out_normals,
    std::vector<glm::vec3> & out_centers,

    std::vector<std::string> & out_material_name,
    std::string & out_mtllib
  );

#endif

// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>  //rand
#include <chrono>
#include <vector>
#include <cstring>
#include <sstream>


#include <glm/glm.hpp>
//using namespace glm;
using namespace std;
typedef std::chrono::time_point<std::chrono::high_resolution_clock> timepoint;

#include "common/tinyply.h"
using namespace tinyply;

#include "MTLplyloader.hpp"


inline double difference_millis(timepoint start, timepoint end)
{
    return (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

std::chrono::high_resolution_clock c;

inline std::chrono::time_point<std::chrono::high_resolution_clock> now()
{
    return c.now();
}

struct float3 { float x, y, z; };
struct float2 {int u, v;};
struct int3 {int a, b, c;};


bool loadPLY(
    const char * path,
    std::vector<glm::vec3> & out_vertices,
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals
){

    std::string path_name = path;
    
    std::ifstream ss(path_name, std::ios::binary);
    if (ss.fail())
    {
        throw std::runtime_error("failed to open " + path_name);
    }

    PlyFile file;
    file.parse_header(ss);
    for (auto c : file.get_comments()) std::cout << "Comment: " << c << std::endl;
    for (auto e : file.get_elements())
    {
        std::cout << "element - " << e.name << " (" << e.size << ")" << std::endl;
        for (auto p : e.properties)
        {
            std::cout << "\tproperty - " << p.name << " (" << tinyply::PropertyTable[p.propertyType].str << ")" << std::endl;
        }
    }
    // Tinyply 2.0 treats incoming data as untyped byte buffers. It's now
    // up to users to treat this data as they wish. See below for examples.
    std::shared_ptr<PlyData> vertices, normals, colors, uvs, faces, texcoords;

    // The header information can be used to programmatically extract properties on elements
    // known to exist in the file header prior to reading the data. For brevity of this sample, properties 
    // like vertex position are hard-coded: 
    try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { colors = file.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { uvs = file.request_properties_from_element("vertex", { "texture_u", "texture_v" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { faces = file.request_properties_from_element("face", { "vertex_indices" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { texcoords = file.request_properties_from_element("face", { "texcoord" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    timepoint before = now();
    file.read(ss);
    timepoint after = now();

    // Good place to put a breakpoint!
    std::cout << "Parsing took " << difference_millis(before, after) << " ms: " << std::endl;
    if (vertices) std::cout << "\tRead " << vertices->count << " total vertices "<< std::endl;
    if (normals) std::cout << "\tRead " << normals->count << " total vertex normals " << std::endl;
    if (colors) std::cout << "\tRead " << colors->count << " total vertex colors "<< std::endl;
    if (uvs) std::cout << "\tRead " << uvs->count << " total vertex uvs" << std::endl;
    if (faces) std::cout << "\tRead " << faces->count << " total faces (triangles) " << std::endl;
    if (texcoords) std::cout << "\tRead " << texcoords->count << " total texcoords " << std::endl;

    std::vector<float3> v_vertices(vertices->count);
    std::vector<float3> v_normals(normals->count);
    std::vector<float2> v_uvs(uvs->count);
    std::vector<int3>   v_faces(faces->count);
    
    // Example: type 'conversion' to your own native types - Option A
    const size_t numVerticesBytes = vertices->buffer.size_bytes();
    const size_t numNormalBytes = normals->buffer.size_bytes();
    const size_t numTextureBytes = uvs->buffer.size_bytes();
    const size_t numFaceBytes = faces->buffer.size_bytes();

    std::memcpy(v_vertices.data(), vertices->buffer.get(), numVerticesBytes);
    std::memcpy(v_normals.data(), normals->buffer.get(), numNormalBytes);
    std::memcpy(v_uvs.data(), uvs->buffer.get(), numTextureBytes);
    std::memcpy(v_faces.data(), faces->buffer.get(), numFaceBytes);

    printf("Vertices bytes %d normal bytes %d texture bytes %d face bytes %d\n", numVerticesBytes, numNormalBytes, numTextureBytes, numFaceBytes);
    float3 temp_vertice;
    float3 temp_normal;
    float2 temp_uv;
    int3 face_i;

    for (unsigned int i = 0; i < faces->count; i++) {
        face_i = v_faces[i];
        //printf("Face coordinates %d/%d (%d %d %d)/%d \n", i, faces->count, face_i.a - 1, face_i.b - 1, face_i.c - 1, vertices->count);
        temp_vertice = v_vertices[face_i.a - 1];
        temp_normal  = v_normals[face_i.a - 1];
        temp_uv      = v_uvs[face_i.a - 1];
        out_vertices.push_back(glm::vec3(temp_vertice.x, temp_vertice.y, temp_vertice.z));
        out_normals.push_back(glm::vec3(temp_normal.x, temp_normal.y, temp_normal.z));
        out_uvs.push_back(glm::vec2(temp_uv.u, temp_uv.v));
        //printf("Vertex a (%f %f %f) normal (%f %f %f) uv (%f %f)\n", temp_vertice.x, temp_vertice.y, temp_vertice.z, temp_normal.x, temp_normal.y, temp_normal.z, temp_uv.u, temp_uv.v);

        temp_vertice = v_vertices[face_i.b - 1];
        temp_normal  = v_normals[face_i.b - 1];
        temp_uv      = v_uvs[face_i.b - 1];
        out_vertices.push_back(glm::vec3(temp_vertice.x, temp_vertice.y, temp_vertice.z));
        out_normals.push_back(glm::vec3(temp_normal.x, temp_normal.y, temp_normal.z));
        out_uvs.push_back(glm::vec2(temp_uv.u, temp_uv.v));
        //printf("Vertex b (%f %f %f) normal (%f %f %f) uv (%f %f)\n", temp_vertice.x, temp_vertice.y, temp_vertice.z, temp_normal.x, temp_normal.y, temp_normal.z, temp_uv.u, temp_uv.v);

        temp_vertice = v_vertices[face_i.c - 1];
        temp_normal  = v_normals[face_i.c - 1];
        temp_uv      = v_uvs[face_i.c - 1];
        out_vertices.push_back(glm::vec3(temp_vertice.x, temp_vertice.y, temp_vertice.z));
        out_normals.push_back(glm::vec3(temp_normal.x, temp_normal.y, temp_normal.z));
        out_uvs.push_back(glm::vec2(temp_uv.u, temp_uv.v));
        //printf("Vertex c (%f %f %f) normal (%f %f %f) uv (%f %f)\n", temp_vertice.x, temp_vertice.y, temp_vertice.z, temp_normal.x, temp_normal.y, temp_normal.z, temp_uv.u, temp_uv.v);
    }

    return true;

}


/*
bool loadPLY_MTL(
    const char * path,
    std::vector<std::vector<glm::vec3>> & out_vertices,
    std::vector<std::vector<glm::vec2>> & out_uvs,
    std::vector<std::vector<glm::vec3>> & out_normals,
    std::vector<std::string> & out_material_name,
    std::string & out_mtllib
){
    printf("Loading OBJ file %s...\n", path);


    std::ifstream ss(path, std::ios::binary);
    if (ss.fail())
    {
        throw std::runtime_error("failed to open " + name_ply);
    }

    PlyFile file;
    file.parse_header(ss);
    for (auto c : file.get_comments()) std::cout << "Comment: " << c << std::endl;
    for (auto e : file.get_elements())
    {
        std::cout << "element - " << e.name << " (" << e.size << ")" << std::endl;
        for (auto p : e.properties)
        {
            std::cout << "\tproperty - " << p.name << " (" << tinyply::PropertyTable[p.propertyType].str << ")" << std::endl;
        }
    }
    // Tinyply 2.0 treats incoming data as untyped byte buffers. It's now
    // up to users to treat this data as they wish. See below for examples.
    std::shared_ptr<PlyData> vertices, normals, colors, faces, texcoords;

    // The header information can be used to programmatically extract properties on elements
    // known to exist in the file header prior to reading the data. For brevity of this sample, properties 
    // like vertex position are hard-coded: 
    try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { colors = file.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { faces = file.request_properties_from_element("face", { "vertex_indices" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { texcoords = file.request_properties_from_element("face", { "texcoord" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    timepoint before = now();
    file.read(ss);
    timepoint after = now();

    // Good place to put a breakpoint!
    std::cout << "Parsing took " << difference_millis(before, after) << " ms: " << std::endl;
    if (vertices) std::cout << "\tRead " << vertices->count << " total vertices "<< std::endl;
    if (normals) std::cout << "\tRead " << normals->count << " total vertex normals " << std::endl;
    if (colors) std::cout << "\tRead " << colors->count << " total vertex colors "<< std::endl;
    if (faces) std::cout << "\tRead " << faces->count << " total faces (triangles) " << std::endl;
    if (texcoords) std::cout << "\tRead " << texcoords->count << " total texcoords " << std::endl;

    // Example: type 'conversion' to your own native types - Option A
    struct float3 { float x, y, z; };
    struct int2 {int u, v;};
    struct int4 {int n, a, b, c;};

    std::vector<float3> v_vertices(vertices->count);
    std::memcpy(v_vertices.data(), vertices->buffer.get(), vertices->buffer.size_bytes());

    std::vector<float3> v_normals(normals->count);
    std::memcpy(v_normals.data(), normals->buffer.get(), normals->buffer.size_bytes());

    std::vector<int2> v_textures(texcoords->count);
    std::memcpy(v_textures.data(), texcoords->buffer.get(), texcoords->buffer.size_bytes());

    std::vector<int4> v_faces(faces->count);
    std::memcpy(v_faces.data(), texcoords->buffer.get(), texcoords->buffer.size_bytes());


    std::vector<std::vector<unsigned int>> vertexIndices, uvIndices, normalIndices;
    std::vector<glm::vec3> temp_vertices;
    std::vector<glm::vec2> temp_uvs;
    std::vector<glm::vec3> temp_normals;

    std::vector<unsigned int> temp_vertexIndices;
    std::vector<unsigned int> temp_uvIndices;
    std::vector<unsigned int> temp_normalIndices;



    
    vertexIndices.push_back(temp_vertexIndices);
    uvIndices.push_back(temp_uvIndices);
    normalIndices.push_back(temp_normalIndices);
    temp_vertexIndices.clear();
    temp_uvIndices.clear();
    temp_normalIndices.clear();

    
    
    printf("%s: size of temp vertices %lu, vertex indices %lu out vertices %lu\n", path, temp_vertices.size(), vertexIndices.size(), out_vertices.size());
    return true;
}
*/
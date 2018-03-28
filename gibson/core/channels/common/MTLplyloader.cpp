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

#include "common/picojson.h"
#include "common/MTLplyloader.hpp"

struct float3 { float x, y, z; };
struct float2 {int u, v;};
struct int3 {int a, b, c;};


inline double difference_millis(timepoint start, timepoint end)
{
    return (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

std::chrono::high_resolution_clock c;

inline std::chrono::time_point<std::chrono::high_resolution_clock> now()
{
    return c.now();
}

bool loadPLY(
    const char * path,
    std::vector<glm::vec3> & out_vertices,
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals,
    std::vector<int3> & out_faces,
    int & num_vertices
){

    std::string path_name = path;
    path_name = path_name + "/semantic.ply";
    
    std::ifstream ss(path_name, std::ios::binary);
    if (ss.fail())
    {
        throw std::runtime_error("failed to open " + path_name);
    }

    PlyFile file;
    file.parse_header(ss);
    /*
    for (auto c : file.get_comments()) std::cout << "Comment: " << c << std::endl;
    for (auto e : file.get_elements())
    {
        std::cout << "element - " << e.name << " (" << e.size << ")" << std::endl;
        for (auto p : e.properties)
        {
            std::cout << "\tproperty - " << p.name << " (" << tinyply::PropertyTable[p.propertyType].str << ")" << std::endl;
        }
    }
    */
    std::shared_ptr<PlyData> vertices, normals, colors, uvs, faces, texcoords;

    // The header information can be used to programmatically extract properties on elements
    // known to exist in the file header prior to reading the data. For brevity of this sample, properties 
    // like vertex position are hard-coded: 
    try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
    catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl; */}

    try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
    catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl; */}

    try { colors = file.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" }); }
    catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl; */}

    try { uvs = file.request_properties_from_element("vertex", { "texture_u", "texture_v" }); }
    catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl; */}

    try { faces = file.request_properties_from_element("face", { "vertex_indices" }); }
    catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl; */}

    try { texcoords = file.request_properties_from_element("face", { "texcoord" }); }
    catch (const std::exception & e) { /*std::cerr << "tinyply exception: " << e.what() << std::endl; */}

    timepoint before = now();
    file.read(ss);
    timepoint after = now();

    // Good place to put a breakpoint!
    std::cout << "Parsing took " << difference_millis(before, after) << " ms: " << std::endl;
    if (vertices)   std::cout << "\tRead " << vertices->count << " total vertices "<< std::endl;
    if (normals)    std::cout << "\tRead " << normals->count << " total vertex normals " << std::endl;
    if (colors)     std::cout << "\tRead " << colors->count << " total vertex colors "<< std::endl;
    if (uvs)        std::cout << "\tRead " << uvs->count << " total vertex uvs" << std::endl;
    if (faces)      std::cout << "\tRead " << faces->count << " total faces (triangles) " << std::endl;
    if (texcoords)  std::cout << "\tRead " << texcoords->count << " total texcoords " << std::endl;

    num_vertices = vertices->count;
    std::vector<float3> v_vertices;
    std::vector<int3>   v_faces;
    std::vector<float3> v_normals;
    std::vector<float2> v_uvs;

    v_vertices  = std::vector<float3>(vertices->count);
    v_faces     =  std::vector<int3>(faces->count);
    if (normals) v_normals = std::vector<float3>(normals->count);
    if (uvs)     v_uvs     = std::vector<float2>(uvs->count);
    
    // Example: type 'conversion' to your own native types - Option A
    size_t numVerticesBytes;
    size_t numNormalBytes;
    size_t numFaceBytes;
    size_t numTextureBytes;
    numVerticesBytes = vertices->buffer.size_bytes();
    numNormalBytes = normals->buffer.size_bytes();
    if (normals) numFaceBytes = faces->buffer.size_bytes();
    if (uvs)     numTextureBytes = uvs->buffer.size_bytes();
    
    std::memcpy(v_vertices.data(), vertices->buffer.get(), numVerticesBytes);
    std::memcpy(v_faces.data(), faces->buffer.get(), numFaceBytes);
    if (normals) std::memcpy(v_normals.data(), normals->buffer.get(), numNormalBytes);
    if (uvs) std::memcpy(v_uvs.data(), uvs->buffer.get(), numTextureBytes);
    
    float3 temp_vertice;
    float3 temp_normal;
    float2 temp_uv;
    int3 face_i;


    for (unsigned int i = 0; i < faces->count; i++)
        out_faces.push_back(v_faces[i]);

    for (unsigned int i = 0; i < vertices->count; i++) {
        temp_vertice = v_vertices[i];
        out_vertices.push_back(glm::vec3(temp_vertice.x, temp_vertice.y, temp_vertice.z));
        if (normals) {
            temp_normal  = v_normals[i];
            out_normals.push_back(glm::vec3(temp_normal.x, temp_normal.y, temp_normal.z));
        }
        if (uvs) {
            temp_uv      = v_uvs[i];
            out_uvs.push_back(glm::vec2(temp_uv.u, temp_uv.v));
        }
    }
    /*
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
    */

    return true;

};


/* Load semantic JSON file for Matterport3D dataset
 * Require: semantic.fsegs.json, semantic.semseg.json
 * Output:
 *      out_label_id: list of semantic segment label 
 *      out_segment_id: list of segmentic segment id (start from 0, incremental)
 *      out_face_indices: list of semantic groups, each group contains all faces of the segment
 * Example:
 *      # 0th semantic component
 *      out_label_id[0] = 1                  # Semantic id is 1 
 *      out_segment_id[0] = 0                # Semantic label index is 0 (start from 0)
 *      out_face_indices[0] = [0, 1, 2]      # Face index 0, 1, 2
 */
bool loadJSONtextures (
    std::string jsonpath, 
    std::vector<int> & out_label_id, 
    std::vector<int> & out_segment_id,
    std::vector<std::vector<int>> & out_face_indices
) {
    std::string imagePath;
    std::string fsegs_json = jsonpath + "/semantic.fsegs.json";
    std::ifstream ss(fsegs_json, std::ios::binary);
    if (ss.fail())
        throw std::runtime_error("failed to open " + fsegs_json);

    std::vector<int> segid_list;
    std::vector<std::vector<int>> segid_to_index;
    std::vector<int> index_to_segment_id;
    std::vector<int> index_to_segment_label_id;

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
            index_to_segment_id.push_back(0);
            index_to_segment_label_id.push_back(0);
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

    printf("Finished loading indexes: %lu\n", segid_list.size());

    std::string ssegs_json = jsonpath + "/semantic.semseg.json";
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
            int segment_id = stoi(obj_it["id"].to_str());
            int label_id = stoi(obj_it["label_index"].to_str());
            // Output to texture name
            out_label_id.push_back(label_id);
            out_segment_id.push_back(segment_id);
            std::vector<int> face_index;

            picojson::value seg_val = obj_it["segments"];
            picojson::array seg_arr_i;
            if (seg_val.is<picojson::array>()) {
                seg_arr_i = seg_val.get<picojson::array>();
                for (unsigned int i = 0; i < seg_arr_i.size(); i++) {
                    int segid = (stoi(seg_arr_i[i].to_str()));
                    for (unsigned int j = 0; j < segid_to_index[segid].size(); j++) {
                        int index = segid_to_index[segid][j];
                        index_to_segment_id[index]    = segment_id;
                        index_to_segment_label_id[index] = label_id;

                        face_index.push_back(index);
                    }
                }
            }
            // Output the associated vertex indices for this texture name
            out_face_indices.push_back(face_index);
        }
        delete[] buffer;
    }

    return true;
};


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
){
    //printf("Loading OBJ file %s...\n", path);

    std::vector<glm::vec3> all_vertices;
    std::vector<glm::vec2> all_uvs;
    std::vector<glm::vec3> all_normals;
    std::vector<int3> all_faces;

    std::vector<int> label_index;                       // Actual id in Matterport3D
    std::vector<int> segment_index; 
    std::vector<std::vector<int>> face_indices;         // Auxiliary index
    loadPLY(path, all_vertices, all_uvs, all_normals, all_faces, num_vertices);
    loadJSONtextures(path, label_index, segment_index, face_indices);

    bool has_uvs = all_uvs.size() > 0;
    bool has_normals = all_normals.size() > 0;

    std::cout << "Loaded many faces " << all_faces.size() << std::endl;
    if (! all_faces.size() == face_indices.size()) {
        std::cerr << "Your ply object file (" << all_faces.size() << ") does not match with JSON file (" << face_indices.size() << ")." << std::endl;
        exit(2);
    }

    for (unsigned int i = 0; i < label_index.size(); i++) {
        out_material_id.push_back(label_index[i]);
        std::vector<int> faces = face_indices[i];

        std::vector<glm::vec3> curr_vertices;
        std::vector<glm::vec2> curr_uvs;
        std::vector<glm::vec3> curr_normals;
        glm::vec3 center = glm::vec3(0.0);
        for (unsigned int j = 0; j < faces.size(); j++) {
            int face_index = faces[j];
            int3 curr_face = all_faces[face_index];
            curr_vertices.push_back(all_vertices[curr_face.a]);
            curr_vertices.push_back(all_vertices[curr_face.b]);
            curr_vertices.push_back(all_vertices[curr_face.c]);
            center += (all_vertices[curr_face.a] + all_vertices[curr_face.b] + all_vertices[curr_face.c]) / 3.0f;
            if (has_uvs) {
                curr_uvs.push_back(all_uvs[curr_face.a]);
                curr_uvs.push_back(all_uvs[curr_face.b]);
                curr_uvs.push_back(all_uvs[curr_face.c]);
            }
            if (has_normals) {
                curr_normals.push_back(all_normals[curr_face.a]);
                curr_normals.push_back(all_normals[curr_face.b]);
                curr_normals.push_back(all_normals[curr_face.c]);
            }
        }
        center /= faces.size();

        out_centers.push_back(center);
        out_vertices.push_back(curr_vertices);
        if (has_uvs) out_uvs.push_back(curr_uvs);
        if (has_normals) out_normals.push_back(curr_normals);
    }

    return true;
}

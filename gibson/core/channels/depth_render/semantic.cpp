// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>  //rand
#include <chrono>
#include "boost/multi_array.hpp"
#include "boost/timer.hpp"


#include  <glad/egl.h>
#include  <glad/gl.h>
#include "lodepng.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

using namespace std;

#include <common/shader.hpp>
#include <common/texture.hpp>
#include <common/objloader.hpp>
#include <common/vboindexer.hpp>
#include <common/cmdline.h>
#include <common/controls.hpp>
#include <common/semantic_color.hpp>

#include <common/MTLobjloader.hpp>
#include <common/MTLplyloader.hpp>
#include <common/MTLtexture.hpp>
#include <zmq.hpp>

#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>
#define sleep(n)    Sleep(n)
#endif


int main( int argc, char * argv[] )
{

    cmdline::parser cmdp;
    cmdp.add<std::string>("modelpath", 'd', "data model directory", true, "");
    cmdp.add<int>("Semantic Source", 'r', "Semantic data source", false, 1);
    cmdp.add<int>("Port", 'p', "Semantic loading port", false, 5055);
    
    cmdp.parse_check(argc, argv);

    std::string model_path = cmdp.get<std::string>("modelpath");
    int port = cmdp.get<int>("Port");
    int semantic_src = cmdp.get<int>("Semantic Source");
    int ply;
    
    std::string name_obj = model_path + "/mesh.obj";
    name_obj = model_path + "/semantic.obj";
    if (semantic_src == 1) ply = 0;
    if (semantic_src == 2) ply = 1;
    
    std::vector<std::vector<glm::vec3>> mtl_vertices;
    std::vector<std::vector<glm::vec2>> mtl_uvs;
    std::vector<std::vector<glm::vec3>> mtl_normals;
    std::vector<glm::vec3> mtl_sem_centers;
    std::vector<std::string> material_name;
    std::vector<int> material_id;
    std::string mtllib;

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> normals;
    std::vector<TextureObj> TextObj;
    unsigned int num_layers;

    /* initialize random seed: */
    srand (0);

    bool res;
    int num_vertices;
    if (ply > 0) {
        res = loadPLY_MTL(model_path.c_str(), mtl_vertices, mtl_uvs, mtl_normals, mtl_sem_centers, material_id, mtllib, num_vertices);
        printf("From ply loaded total of %d vertices\n", num_vertices);
    } else {
        res = loadOBJ_MTL(name_obj.c_str(), mtl_vertices, mtl_uvs, mtl_normals, mtl_sem_centers, material_name, mtllib);
        for (int i = 0; i < 20; i++) {
            printf("Loaded semantic center %f, %f, %f\n", mtl_sem_centers[i].x, mtl_sem_centers[i].y, mtl_sem_centers[i].z);
        }
    }
    if (res == false) { printf("Was not able to load the semantic.obj file.\n"); exit(-1); }
    else { printf("Semantic.obj file was loaded with success.\n"); }

    // Load the textures
    std::string mtl_path = model_path + "/" + mtllib;
    bool MTL_loaded;
    if (ply > 0) {
        mtl_path = model_path;
        MTL_loaded = loadPLYtextures(TextObj, material_id);
    } else {
        MTL_loaded = loadMTLtextures(mtl_path, TextObj, material_name);    
    }
    if (MTL_loaded == false) { printf("Was not able to load textures\n"); exit(-1); }
    else { printf("Texture file was loaded with success, total: %lu\n", TextObj.size()); }

    // Read our .obj file
    // Note: use unsigned int because of too many indices
    std::vector<unsigned int> indices;
    std::vector<glm::vec3> indexed_vertices;
    std::vector<glm::vec2> indexed_uvs;
    std::vector<glm::vec3> indexed_normals;
    std::vector<glm::vec2> indexed_semantics;

    /*
    indexVBO_MTL(mtl_vertices, mtl_uvs, mtl_normals, indices, indexed_vertices, indexed_uvs, indexed_normals, indexed_semantics);
    std::cout << "Finished indexing vertices v " << indexed_vertices.size() << " uvs " << indexed_uvs.size() << " normals " << indexed_normals.size() << " semantics " << indexed_semantics.size() << std::endl;
    std::cout << "Semantics ";
    std::cout << std::endl;
    */

    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REP);
    zmq::message_t request;
    socket.bind ("tcp://127.0.0.1:"  + std::to_string(port));

    //  Wait for next request from client
    socket.recv (&request);
    std::string request_str = std::string(static_cast<char*>(request.data()), request.size());


    int dim = 3;
    int message_sz = mtl_sem_centers.size()*sizeof(float)*dim;
    zmq::message_t reply (message_sz);

    float * reply_data_handle = (float*)reply.data();
    for (int i = 0; i < mtl_sem_centers.size(); i++) {
        for (int k = 0; k < dim; k++) {
            int offset = k;
            float tmp_float = mtl_sem_centers[i][k];
            reply_data_handle[offset + i * dim] = tmp_float;
        }  
    }

    socket.send (reply);
    
    return 0;
}

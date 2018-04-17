#include <vector>
#include <stdio.h>
#include <string>
#include <cstring>
#include <sstream>

#include <glm/glm.hpp>

#include "MTLobjloader.hpp"

// Very, VERY simple OBJ loader.
// Here is a short list of features a real function would provide :
// - Binary files. Reading a model should be just a few memcpy's away, not parsing a file at runtime. In short : OBJ is not very great.
// - Animations & bones (includes bones weights)
// - Multiple UVs
// - All attributes should be optional, not "forced"
// - More stable. Change a line in the OBJ file and it crashes.
// - More secure. Change another line and you can inject code.
// - Loading from memory, stream, etc

bool loadOBJ_MTL(
    const char * path,
    std::vector<std::vector<glm::vec3>> & out_vertices,
    std::vector<std::vector<glm::vec2>> & out_uvs,
    std::vector<std::vector<glm::vec3>> & out_normals,
    std::vector<glm::vec3> & out_centers,

    std::vector<std::string> & out_material_name,
    std::string & out_mtllib
){
    printf("Loading OBJ file %s...\n", path);

    std::vector<std::vector<unsigned int>> vertexIndices, uvIndices, normalIndices;
    std::vector<glm::vec3> temp_vertices;
    std::vector<glm::vec2> temp_uvs;
    std::vector<glm::vec3> temp_normals;

    std::vector<unsigned int> temp_vertexIndices;
    std::vector<unsigned int> temp_uvIndices;
    std::vector<unsigned int> temp_normalIndices;


    FILE * file = fopen(path, "r");
    if( file == NULL ){
        printf("Impossible to open the file ! Are you in the right path ? Given path: %s\n", path);
        getchar();
        return false;
    }

    unsigned int v_count = 0;
    unsigned int mtl_count = 0;
    std::string last_mtl_name;

    while( 1 ){

        char lineHeader[128];
        // read the first word of the line
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF)
            break; // EOF = End Of File. Quit the loop.

        // get mtllib name
        if ( strcmp( lineHeader, "mtllib") == 0 ){
            char mtllib;
            fscanf(file, "%s \n", &mtllib);
            out_mtllib = &mtllib;
            //printf("Material Name: %s\n", &mtllib);
        }
        //vertices
        else if ( strcmp( lineHeader, "v" ) == 0 ){
            v_count ++;
            glm::vec3 vertex;
            //fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z );
            // opengl = blender rotate around x at 90
            fscanf(file, "%f %f %f\n", &vertex.x, &vertex.z, &vertex.y);
            vertex.y = -vertex.y;
            temp_vertices.push_back(vertex);
        }
        // UV
        else if ( strcmp( lineHeader, "vt" ) == 0 ){
            glm::vec2 uv;
            fscanf(file, "%f %f\n", &uv.x, &uv.y );
            uv.y = -uv.y; // Invert V coordinate since we will only use DDS texture, which are inverted. Remove if you want to use TGA or BMP loaders.
            temp_uvs.push_back(uv);
        }
        // normal
        else if ( strcmp( lineHeader, "vn" ) == 0 ){
            glm::vec3 normal;
            fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z );
            temp_normals.push_back(normal);
        }
        // material name
        else if ( strcmp( lineHeader, "usemtl" ) == 0 ){
            char material_name;
            fscanf(file, "%s\n", &material_name );
            std::stringstream ss;
            std::string tempmat2;
            ss << &material_name;
            ss >> tempmat2;
            //printf("material name: %s, length: %i\n", tempmat2.c_str(), tempmat2.length());
            out_material_name.push_back(tempmat2.c_str());
            if (mtl_count > 0) {
                vertexIndices.push_back(temp_vertexIndices);
                uvIndices.push_back(temp_uvIndices);
                normalIndices.push_back(temp_normalIndices);
                
                temp_vertexIndices.clear();
                temp_uvIndices.clear();
                temp_normalIndices.clear();
            } 
            mtl_count ++;
        }
        // face
        else if ( mtl_count > 0 && strcmp( lineHeader, "f" ) == 0 ){
            std::string vertex1, vertex2, vertex3;
            unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
            char stringBuffer[500];
            fgets(stringBuffer, 500, file);
            //int matches = fscanf(file, "%u/%u/%u %u/%u/%u %u/%u/%u\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2] );
            int matches = sscanf(stringBuffer, "%u/%u/%u %u/%u/%u %u/%u/%u\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2] );
            bool f_3_format = (matches == 9);
            bool f_2_format = true;
            if (! f_3_format) {
                // .obj file has `f v1/uv1 v2/uv2 v3/uv3` format
                int matches = sscanf(stringBuffer, " %u/%u %u/%u %u/%u\n", &vertexIndex[0], &uvIndex[0], &vertexIndex[1], &uvIndex[1], &vertexIndex[2], &uvIndex[2] );
                f_2_format = (matches == 6);
                if (! f_2_format) {
                    matches = sscanf(stringBuffer, " %u %u %u\n", &vertexIndex[0], &vertexIndex[1], &vertexIndex[2]);
                    if (matches != 3){
                        printf("File can't be read by our simple parser :-( Try exporting with other options\n");
                        fclose(file);
                        return false;
                    }
                }
            }
            temp_vertexIndices.push_back(vertexIndex[0]);
            temp_vertexIndices.push_back(vertexIndex[1]);
            temp_vertexIndices.push_back(vertexIndex[2]);

            if (f_2_format || f_3_format) {
                temp_uvIndices    .push_back(uvIndex[0]);
                temp_uvIndices    .push_back(uvIndex[1]);
                temp_uvIndices    .push_back(uvIndex[2]);
            }
            if (f_3_format) {
                temp_normalIndices.push_back(normalIndex[0]);
                temp_normalIndices.push_back(normalIndex[1]);
                temp_normalIndices.push_back(normalIndex[2]);
            }
        }
        // other
        else {
            // Probably a comment, eat up the rest of the line
            char stupidBuffer[1000];
            fgets(stupidBuffer, 1000, file);
        }
    }

    vertexIndices.push_back(temp_vertexIndices);
    uvIndices.push_back(temp_uvIndices);
    normalIndices.push_back(temp_normalIndices);
    temp_vertexIndices.clear();
    temp_uvIndices.clear();
    temp_normalIndices.clear();

    // Calculate semantic center position
    for (unsigned int i=0; i<vertexIndices.size(); i++) {
        std::vector<unsigned int> vertexIndex_group = vertexIndices[i];
        glm::vec3 center = glm::vec3(0.0);

        for (unsigned int j=0; j<vertexIndex_group.size();j++) {
            unsigned int vertexIndex = vertexIndex_group[j];
            glm::vec3 vertex = temp_vertices[ vertexIndex-1 ];
            center += vertex / 3.0f;            
        }
        center /= (vertexIndex_group.size() / 3);
        out_centers.push_back(center);
        center = glm::vec3(0.0);
    }

    // For each vertex of each triangle
    for( unsigned int i=0; i<vertexIndices.size(); i++ ){
        std::vector<glm::vec3> temp_out_vertices;
        std::vector<glm::vec2> temp_out_uvs;
        std::vector<glm::vec3> temp_out_normals;

        //printf("Vertex group %d size %d\n", i + 1, vertexIndices.size());
        std::string mat_name = out_material_name[i];
        /*if (mat_name == "wall_65_hallway_7_1") {
            for (unsigned int j = 0; j < vertexIndices[i].size(); j++) {
                printf("%s %d\n", mat_name.c_str(), vertexIndices[i][j]);
            }
        }*/
        for ( unsigned int j=0; j<vertexIndices[i].size(); j++) {
            // Get the indices of its attributes
            unsigned int vertexIndex = vertexIndices[i][j];
            unsigned int uvIndex = -1;

            if (uvIndices.size() > 0 && uvIndices[i].size() > 0)
                uvIndex = uvIndices[i][j];

            unsigned int normalIndex = -1;
            if (normalIndices.size() > 0 && normalIndices[i].size() > 0)
                normalIndex = normalIndices[i][j];

            // Get the attributes thanks to the index
            glm::vec3 vertex = temp_vertices[ vertexIndex-1 ];

            // Put the attributes in buffers
            temp_out_vertices.push_back(vertex);

            if (temp_uvs.size() > 0 && uvIndices.size() > 0) {
                glm::vec2 uv = temp_uvs[ uvIndex-1 ];
                temp_out_uvs     .push_back(uv);
            }

            if (temp_normals.size() > 0 && normalIndices.size() > 0) {
                glm::vec3 normal = temp_normals[ normalIndex-1 ];
                temp_out_normals.push_back(normal);
            }
        }
        out_vertices.push_back(temp_out_vertices);
        if (temp_out_uvs.size() > 0)
            out_uvs.push_back(temp_out_uvs);
        if (temp_out_normals.size() > 0)
            out_normals.push_back(temp_out_normals);
    }

    // construct the temp_normals vector here, using vertex positions and face vertex indices
    // TODO: this is not well-tested yet
    
    if ( out_normals.size() == 0 ) {
        for ( unsigned int i=0; i<out_vertices.size(); i++ ){
            std::vector<glm::vec3> temp_out_normals;
            for ( unsigned int j=0; j<out_vertices[i].size(); j++ ){
                temp_out_normals.push_back(glm::vec3(0.0));
            }
            out_normals.push_back(temp_out_normals);
        }

        std::vector<std::vector<unsigned int>> vertexFaces;
        for ( unsigned int i=0; i<vertexIndices.size(); i++ ){
            std::vector<unsigned int> temp_vertexFaces(out_vertices[i].size());
            std::fill(temp_vertexFaces.begin(), temp_vertexFaces.end(), 0);
            for ( unsigned int j=0; j<vertexIndices[i].size(); j++ ){
                temp_vertexFaces[vertexIndices[i][j]] += 1;
            }
            vertexFaces.push_back(temp_vertexFaces);
        }

        for ( unsigned int i=0; i<vertexIndices.size(); i++ ){
            for ( unsigned int j=0; j<vertexIndices[i].size(); j++ ){
                // make sure vertices are arranged in right hand order
                unsigned int v1 = j;
                unsigned int v2 = v1; // ((v1+1)%3==0) ? (v1-2) : (v1+1);
                unsigned int v3 = v1; //((v2+1)%3==0) ? (v2-2) : (v2+1);

                glm::vec3 edge1 = out_vertices[i][v2] - out_vertices[i][v1];
                glm::vec3 edge2 = out_vertices[i][v3] - out_vertices[i][v2];

                // set normal as cross product
                unsigned int vertexIndex = vertexIndices[i][j];
                glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));
                out_normals[i][vertexIndex-1] += normal / float(vertexFaces[i][vertexIndex-1]);
                //printf("%f %f %f\n", normal[0], normal[1], normal[2]);
            }
        }
    }
    

    // TODO: (hzyjerry) this is a dummy place holder
    if ( out_uvs.size() == 0 ) {
      for ( unsigned int i=0; i<out_vertices.size(); i++ ){
            std::vector<glm::vec2> temp_out_uvs;
            for ( unsigned int j=0; j<out_vertices[i].size(); j++ ){
                temp_out_uvs.push_back(glm::vec2(0.0));
            }
            out_uvs.push_back(temp_out_uvs);
        }
    }
    printf("%s: size of temp vertices %lu, vertex indices %lu out vertices %lu\n", path, temp_vertices.size(), vertexIndices.size(), out_vertices.size());
    fclose(file);
    return true;
}

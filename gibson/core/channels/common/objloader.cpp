#include <vector>
#include <stdio.h>
#include <string>
#include <cstring>
#include <iostream>
#include <glm/glm.hpp>
#include <map>
#include "objloader.hpp"

// Very, VERY simple OBJ loader.
// Here is a short list of features a real function would provide :
// - Binary files. Reading a model should be just a few memcpy's away, not parsing a file at runtime. In short : OBJ is not very great.
// - Animations & bones (includes bones weights)
// - Multiple UVs
// - All attributes should be optional, not "forced"
// - More stable. Change a line in the OBJ file and it crashes.
// - More secure. Change another line and you can inject code.
// - Loading from memory, stream, etc


struct Surface{
    unsigned int v1;
    unsigned int v2;
    unsigned int v3;
    bool operator<(const Surface that) const{
        // Calculate a hash of the surface
        return (92233 * v1 + 92003 * v2 + 90793 * v3) <
        (92233 * that.v1 + 92003 * that.v2 + 90793 * that.v3);
    };
};

bool getIdenticalSurface_fast( 
    Surface & surf,
    std::map<Surface,unsigned int> & SurfaceToOutIndex,
    unsigned int & result
){
    std::map<Surface,unsigned int>::iterator it = SurfaceToOutIndex.find(surf);
    if ( it == SurfaceToOutIndex.end() ){
        return false;
    }else{
        result = it->second;
        // printf("it->second %lu\n", it->second);
        return true;
    }
}


float hash1(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3) {
    double s1 = (53.0 / v1.x + 97.0 / v1.y + 193.0 / v1.z \
        + 389.0 / v2.x + 769.0 / v2.y + 1543.0 / v2.z \
        + 3079.0 / v3.x + 6151.0 / v3.y + 12289.0 / v3.z);
    double s2 = (53.0 * v1.x * v1.x + 97.0 * v1.y * v1.y + 193.0 * v1.z * v1.z \
        + 389.0 * v2.x * v2.x + 769.0 * v2.y * v2.y + 1543.0 * v2.z * v2.z\
        + 3079.0 * v3.x * v3.x + 6151.0 * v3.y * v2.y + 12289.0 * v3.z * v3.z);
    return s1 + s2;
}

float hash2(glm::vec3 v) {
    //double s1 = (53.0 / v.x + 97.0 / v.y + 193.0 / v.z);
    //double s2 = (53.0 * v.x * v.x + 97.0 * v.y * v.y + 193.0 * v.z * v.z);
    //return s1 + s2;
    return (53.0 * v.x + 97.0 * v.y + 193.0 * v.z);   
}


float lessThan(glm::vec3 a1, glm::vec3 a2, glm::vec3 a3, glm::vec3 b1, glm::vec3 b2, glm::vec3 b3) {
    // Calculate less than given current sequence
    double this_hash_1 = hash2(a1);
    double this_hash_2 = hash2(a2);
    double this_hash_3 = hash2(a3);
    double that_hash_1 = hash2(b1);
    double that_hash_2 = hash2(b2);
    double that_hash_3 = hash2(b3);
    double threshold = 0.001;
    if (this_hash_1 + threshold < that_hash_1) return true;
    if (this_hash_1 - threshold > that_hash_1) return false;

    if (this_hash_2 + threshold < that_hash_2) return true;
    if (this_hash_2 - threshold > that_hash_2) return false;

    if (this_hash_3 + threshold < that_hash_3) return true;
    if (this_hash_3 - threshold > that_hash_3) return false;

    return  false;
}

struct VecSurface{
    glm::vec3 v1;
    glm::vec3 v2;
    glm::vec3 v3;
    bool operator<(const VecSurface that) const{
        // Calculate a hash of the surface
        //double this_hash = hash1(v1, v2, v3);
        //double that_hash = hash1(that.v1, that.v2, that.v3);
        return lessThan(v1, v2, v3, that.v1, that.v2, that.v3) or 
               lessThan(v1, v3, v2, that.v1, that.v2, that.v3) or
               lessThan(v2, v1, v3, that.v1, that.v2, that.v3) or
               lessThan(v2, v3, v1, that.v1, that.v2, that.v3) or
               lessThan(v3, v1, v2, that.v1, that.v2, that.v3) or
               lessThan(v3, v2, v1, that.v1, that.v2, that.v3);
    };
};


bool getSimilarSurface_fast( 
    VecSurface & surf,
    std::map<VecSurface,unsigned int> & VecSurfaceToOutIndex,
    unsigned int & result
){
    std::map<VecSurface,unsigned int>::iterator it = VecSurfaceToOutIndex.find(surf);
    if ( it == VecSurfaceToOutIndex.end() ){
        return false;
    }else{
        result = it->second;
        // printf("it->second %lu\n", it->second);
        return true;
    }
}


bool loadOBJ(
    const char * path,
    std::vector<glm::vec3> & out_vertices,
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals
){
    //printf("Loading OBJ file %s...\n", path);

    std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
    std::vector<glm::vec3> temp_vertices;
    std::vector<glm::vec2> temp_uvs;
    std::vector<glm::vec3> temp_normals;

    FILE * file = fopen(path, "r");
    if( file == NULL ){
        printf("Impossible to open the file %s\n", path);
        getchar();
        return false;
    }

    unsigned int v_count = 0;
    std::map<Surface,unsigned int> SurfaceToOutIndex;
    std::vector<Surface> allSurface;

    while( 1 ){

        char lineHeader[128];
        // read the first word of the line
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF)
            break; // EOF = End Of File. Quit the loop.

        // else : parse lineHeader

        if ( strcmp( lineHeader, "v" ) == 0 ){
            v_count ++;
            glm::vec3 vertex;
            //fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z );
            // opengl = blender rotate around x at 90
            fscanf(file, "%f %f %f\n", &vertex.x, &vertex.z, &vertex.y);
            vertex.y = -vertex.y;
            temp_vertices.push_back(vertex);
        }else if ( strcmp( lineHeader, "vt" ) == 0 ){
            glm::vec2 uv;
            fscanf(file, "%f %f\n", &uv.x, &uv.y );
            uv.y = -uv.y; // Invert V coordinate since we will only use DDS texture, which are inverted. Remove if you want to use TGA or BMP loaders.
            temp_uvs.push_back(uv);
        }else if ( strcmp( lineHeader, "vn" ) == 0 ){
            glm::vec3 normal;
            fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z );
            temp_normals.push_back(normal);
        }else if ( strcmp( lineHeader, "f" ) == 0 ){
            std::string vertex1, vertex2, vertex3;
            unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
            char stringBuffer[500];
            fgets(stringBuffer, 500, file);
            //int matches = fscanf(file, "%u/%u/%u %u/%u/%u %u/%u/%u\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2] );
            int matches = sscanf(stringBuffer, "%u/%u/%u %u/%u/%u %u/%u/%u\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2] );
            bool f_3_format = (matches == 9);
            bool f_2_format = true;
            bool f_2_format_normal = true;
            if (! f_3_format) {
                // .obj file has `f v1/uv1 v2/uv2 v3/uv3` format
                int matches = sscanf(stringBuffer, " %u/%u %u/%u %u/%u\n", &vertexIndex[0], &uvIndex[0], &vertexIndex[1], &uvIndex[1], &vertexIndex[2], &uvIndex[2] );
                f_2_format = (matches == 6);
                if (! f_2_format) {
                    int matches = sscanf(stringBuffer, " %u//%u %u//%u %u//%u\n", &vertexIndex[0], &normalIndex[0], &vertexIndex[1], &normalIndex[1], &vertexIndex[2], &normalIndex[2] );
                    f_2_format_normal = (matches == 6);
                    if (! f_2_format_normal) {
                        int matches = sscanf(stringBuffer, " %u %u %u\n", &vertexIndex[0], &vertexIndex[1], &vertexIndex[2]);
                        if (matches != 3){
                            printf("File %s can't be read by our simple parser :-( Try exporting with other options\n", path);
                            fclose(file);
                            return false;
                        }

                    }
                }
            }
            
            Surface surf = {vertexIndex[0],vertexIndex[1], vertexIndex[2]};
            unsigned int index;
            //bool found = getIdenticalSurface_fast(surf, SurfaceToOutIndex, index);

            /*if (found) {
                // (hzyjerry) Important: In this case, do not add in the surface, to avoid z-fighting
                //std::cout << "Found " << std::endl;
                Surface s = allSurface[index];
                vertexIndex[0] = s.v1;
                vertexIndex[1] = s.v2;
                vertexIndex[2] = s.v3;
            } //else {*/
            allSurface.push_back(surf);
            unsigned int newindex = (unsigned int) allSurface.size() - 1;
            SurfaceToOutIndex[ surf ] = newindex;

            vertexIndices.push_back(vertexIndex[0]);
            vertexIndices.push_back(vertexIndex[1]);
            vertexIndices.push_back(vertexIndex[2]);


            if (f_2_format || f_3_format) {
                uvIndices    .push_back(uvIndex[0]);
                uvIndices    .push_back(uvIndex[1]);
                uvIndices    .push_back(uvIndex[2]);
            }
            if (f_3_format || f_2_format_normal) {
                normalIndices.push_back(normalIndex[0]);
                normalIndices.push_back(normalIndex[1]);
                normalIndices.push_back(normalIndex[2]);
            }
            //}

        }else{
            // Probably a comment, eat up the rest of the line
            char stupidBuffer[1000];
            fgets(stupidBuffer, 1000, file);
        }

    }


    std::map<VecSurface,unsigned int> VecSurfaceToOutIndex;
    std::vector<unsigned int> cleanVertexIndices;
    std::vector<VecSurface> allVecSurface;
    unsigned num_similar = 0;

    // Sanitize all surface vertices to remove similar surfaces
    for( unsigned int i=0; i<vertexIndices.size(); i+=3 ){
        // Get the indices of its attributes
        unsigned int index1 = vertexIndices[i];
        unsigned int index2 = vertexIndices[i + 1];
        unsigned int index3 = vertexIndices[i + 2];
        glm::vec3 v1 = temp_vertices[ index1 -1 ];
        glm::vec3 v2 = temp_vertices[ index2 -1 ];
        glm::vec3 v3 = temp_vertices[ index3 -1 ];
        VecSurface vs = {v1, v2, v3};
        unsigned int index;
        //bool found = getSimilarSurface_fast(vs, VecSurfaceToOutIndex, index);
        /*if (found) {
            num_similar += 1;
            
            // To remove duplicated
            //VecSurface that = allVecSurface[index];
            

            // To keep duplicated
            allVecSurface.push_back(vs);
            unsigned int newindex = (unsigned int) allVecSurface.size() - 1;
            VecSurfaceToOutIndex[ vs ] = newindex;

            cleanVertexIndices.push_back(index1);
            cleanVertexIndices.push_back(index2);
            cleanVertexIndices.push_back(index3);
        
        } else {*/
            allVecSurface.push_back(vs);
            unsigned int newindex = (unsigned int) allVecSurface.size() - 1;
            VecSurfaceToOutIndex[ vs ] = newindex;

            cleanVertexIndices.push_back(index1);
            cleanVertexIndices.push_back(index2);
            cleanVertexIndices.push_back(index3);
        //}
    }



    // For each vertex of each triangle
    for( unsigned int i=0; i<cleanVertexIndices.size(); i++ ){

        // Get the indices of its attributes
        unsigned int vertexIndex = cleanVertexIndices[i];
        unsigned int uvIndex = -1;

        if (uvIndices.size() > 0)
            uvIndex = uvIndices[i];

        unsigned int normalIndex = -1;
        if (normalIndices.size() > 0)
            normalIndex = normalIndices[i];


        // Get the attributes thanks to the index
        glm::vec3 vertex = temp_vertices[ vertexIndex-1 ];


        // Put the attributes in buffers
        out_vertices.push_back(vertex);

        if (temp_uvs.size() > 0 && uvIndices.size() > 0) {
            glm::vec2 uv = temp_uvs[ uvIndex-1 ];
            out_uvs     .push_back(uv);
        }

        if (temp_normals.size() > 0 && normalIndices.size() > 0) {
            glm::vec3 normal = temp_normals[ normalIndex-1 ];
            out_normals.push_back(normal);
        }

    }


    // construct the temp_normals vector here, using vertex positions and face vertex indices
    // TODO: this is not well-tested yet
    std::vector<unsigned int> vertexFaces(temp_vertices.size());
    std::fill(vertexFaces.begin(), vertexFaces.end(), 0);
    if ( out_normals.size() == 0 ) {
        for ( unsigned int i=0; i<out_vertices.size(); i++ ){
            out_normals.push_back(glm::vec3(0.0));
        }

        for ( unsigned int i=0; i<cleanVertexIndices.size(); i++ ) {
            vertexFaces[cleanVertexIndices[i]] += 1;
        }


        for ( unsigned int i=0; i<cleanVertexIndices.size(); i++ ){
            // make sure vertices are arranged in right hand order
            unsigned int v1 = i;
            unsigned int v2 = ((v1+1)%3==0) ? (v1-2) : (v1+1);
            unsigned int v3 = ((v2+1)%3==0) ? (v2-2) : (v2+1);

            glm::vec3 edge1 = out_vertices[v2] - out_vertices[v1];
            //glm::vec3 edge2 = out_vertices[v3] - out_vertices[v2];
            glm::vec3 edge2 = out_vertices[v3] - out_vertices[v1];

            // set normal as cross product
            unsigned int vertexIndex = cleanVertexIndices[i];
            glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));
            //std::cout << normal.x << " " << normal.y << " " << normal.z <<  " " << normal.x * normal.x + normal.y * normal.y + normal.z * normal.z << " " << float(vertexFaces[vertexIndex-1]) <<  std::endl;
            out_normals[i] += normal / float(vertexFaces[vertexIndex-1]);
            
            //std::cout << "Writing to " << vertexIndex << std::endl;
            //out_normals[vertexIndex-1] = glm::vec3(1, 0, 0);
        }

        // Renormalize all the normal vectors
        for (unsigned int i=0; i<out_normals.size(); i++) {
            glm::vec3 norm = out_normals[i];
            norm = glm::normalize(norm);
            out_normals[i] = glm::normalize(out_normals[i]);
        }        
    }


    // TODO: (hzyjerry) this is a dummy place holder
    if ( out_uvs.size() == 0 ) {
        for ( unsigned int i=0; i<out_vertices.size(); i++ ){
            out_uvs.push_back(glm::vec2(0.0));
        }
    }
    //printf("size of temp vertices %lu, vertex indices %lu out vertices %lu\n", temp_vertices.size(), vertexIndices.size(), out_vertices.size());
    fclose(file);
    return true;
}


#ifdef USE_ASSIMP // don't use this #define, it's only for me (it AssImp fails to compile on your machine, at least all the other tutorials still work)

// Include AssImp
#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags

bool loadAssImp(
    const char * path,
    std::vector<unsigned short> & indices,
    std::vector<glm::vec3> & vertices,
    std::vector<glm::vec2> & uvs,
    std::vector<glm::vec3> & normals
){

    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(path, 0/*aiProcess_JoinIdenticalVertices | aiProcess_SortByPType*/);
    if( !scene) {
        fprintf( stderr, importer.GetErrorString());
        getchar();
        return false;
    }
    const aiMesh* mesh = scene->mMeshes[0]; // In this simple example code we always use the 1rst mesh (in OBJ files there is often only one anyway)

    // Fill vertices positions
    vertices.reserve(mesh->mNumVertices);
    for(unsigned int i=0; i<mesh->mNumVertices; i++){
        aiVector3D pos = mesh->mVertices[i];
        vertices.push_back(glm::vec3(pos.x, pos.y, pos.z));
    }

    // Fill vertices texture coordinates
    uvs.reserve(mesh->mNumVertices);
    for(unsigned int i=0; i<mesh->mNumVertices; i++){
        aiVector3D UVW = mesh->mTextureCoords[0][i]; // Assume only 1 set of UV coords; AssImp supports 8 UV sets.
        uvs.push_back(glm::vec2(UVW.x, UVW.y));
    }

    // Fill vertices normals
    normals.reserve(mesh->mNumVertices);
    for(unsigned int i=0; i<mesh->mNumVertices; i++){
        aiVector3D n = mesh->mNormals[i];
        normals.push_back(glm::vec3(n.x, n.y, n.z));
    }


    // Fill face indices
    indices.reserve(3*mesh->mNumFaces);
    for (unsigned int i=0; i<mesh->mNumFaces; i++){
        // Assume the model has only triangles.
        indices.push_back(mesh->mFaces[i].mIndices[0]);
        indices.push_back(mesh->mFaces[i].mIndices[1]);
        indices.push_back(mesh->mFaces[i].mIndices[2]);
    }

    // The "scene" pointer will be deleted automatically by "importer"
    return true;
}

#endif

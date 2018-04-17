#include <vector>
#include <map>
#include <stdio.h>
#include <glm/glm.hpp>
#include <iostream>
#include "vboindexer.hpp"
#include <string.h> // for memcmp


// Returns true iif v1 can be considered equal to v2
bool is_near(float v1, float v2){
    return fabs( v1-v2 ) < 0.01f;
}

// Searches through all already-exported vertices
// for a similar one.
// Similar = same position + same UVs + same normal
bool getSimilarVertexIndex( 
    glm::vec3 & in_vertex,
    glm::vec2 & in_uv,
    glm::vec3 & in_normal,
    std::vector<glm::vec3> & out_vertices,
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals,
    unsigned int & result
){
    // Lame linear search
    for ( unsigned int i=0; i<out_vertices.size(); i++ ){
        if (
            is_near( in_vertex.x , out_vertices[i].x ) &&
            is_near( in_vertex.y , out_vertices[i].y ) &&
            is_near( in_vertex.z , out_vertices[i].z ) &&
            is_near( in_uv.x     , out_uvs     [i].x ) &&
            is_near( in_uv.y     , out_uvs     [i].y ) &&
            is_near( in_normal.x , out_normals [i].x ) &&
            is_near( in_normal.y , out_normals [i].y ) &&
            is_near( in_normal.z , out_normals [i].z )
        ){
            result = i;
            return true;
        }
    }
    // No other vertex could be used instead.
    // Looks like we'll have to add it to the VBO.
    return false;
}

void indexVBO_slow(
    std::vector<glm::vec3> & in_vertices,
    std::vector<glm::vec2> & in_uvs,
    std::vector<glm::vec3> & in_normals,

    std::vector<unsigned int> & out_indices,
    std::vector<glm::vec3> & out_vertices,
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals
){
    // For each input vertex
    for ( unsigned int i=0; i<in_vertices.size(); i++ ){

        // Try to find a similar vertex in out_XXXX
        unsigned int index;
        bool found = getSimilarVertexIndex(in_vertices[i], in_uvs[i], in_normals[i],     out_vertices, out_uvs, out_normals, index);

        if ( found ){ // A similar vertex is already in the VBO, use it instead !
            out_indices .push_back( index );
        }else{ // If not, it needs to be added in the output data.
            out_vertices.push_back( in_vertices[i]);
            out_uvs     .push_back( in_uvs[i]);
            out_normals .push_back( in_normals[i]);
            out_indices .push_back( (unsigned int)out_vertices.size() - 1 );
        }
    }
}

struct PackedVertex{
    glm::vec3 position;
    glm::vec2 uv;
    glm::vec3 normal;
    bool operator<(const PackedVertex that) const{
        return memcmp((void*)this, (void*)&that, sizeof(PackedVertex))>0;
    };
};

bool getSimilarVertexIndex_fast( 
    PackedVertex & packed,
    std::map<PackedVertex,unsigned int> & VertexToOutIndex,
    unsigned int & result
){
    std::map<PackedVertex,unsigned int>::iterator it = VertexToOutIndex.find(packed);
    if ( it == VertexToOutIndex.end() ){
        return false;
    }else{
        result = it->second;
        // printf("it->second %lu\n", it->second);
        return true;
    }
}

void indexVBO(
    std::vector<glm::vec3> & in_vertices,
    std::vector<glm::vec2> & in_uvs,
    std::vector<glm::vec3> & in_normals,

    std::vector<unsigned int> & out_indices,
    std::vector<glm::vec3> & out_vertices,
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals
){
    std::map<PackedVertex,unsigned int> VertexToOutIndex;

    // For each input vertex
    for ( unsigned int i=0; i<in_vertices.size(); i++ ){

        PackedVertex packed = {in_vertices[i], in_uvs[i], in_normals[i]};

        // Try to find a similar vertex in out_XXXX
        unsigned int index;
        bool found = getSimilarVertexIndex_fast( packed, VertexToOutIndex, index);

        if ( found ){ // A similar vertex is already in the VBO, use it instead !
            out_indices .push_back( index );
            // printf("found bef %lu %lu\n", i, index);
        }else{ // If not, it needs to be added in the output data.
            out_vertices.push_back( in_vertices[i]);
            out_uvs     .push_back( in_uvs[i]);
            out_normals .push_back( in_normals[i]);
            unsigned int newindex = (unsigned int)out_vertices.size() - 1;
            out_indices .push_back( newindex );
            VertexToOutIndex[ packed ] = newindex;
        }
    }
    // printf("Size of output vertex vector %lu\n", out_vertices.size());
}

// TODO: Inplement index_SEM from load MTL
void indexVBO_MTL(
    std::vector<std::vector<glm::vec3>> & mtl_vertices,
    std::vector<std::vector<glm::vec2>> & mtl_uvs,
    std::vector<std::vector<glm::vec3>> & mtl_normals,

    std::vector<unsigned int> & out_indices,
    std::vector<glm::vec3> & out_vertices,
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals,
    std::vector<glm::vec2> & out_semantics
) {
    std::map<PackedVertex,unsigned int> VertexToOutIndex;

    // In group j
    bool use_uvs = mtl_uvs.size() > 0;
    bool use_normals = mtl_normals.size() > 0;
    if (!((use_uvs && mtl_vertices.size() == mtl_vertices.size()) || 
          (use_normals && mtl_normals.size() == mtl_normals.size()))) {
        printf("VBO Indexer error: size missmatch. Vertex (%lu) uvs (%lu) normals (%lu)\n", mtl_vertices.size(), mtl_uvs.size(), mtl_normals.size());
    }
    std::vector<glm::vec3> group_vertices;
    std::vector<glm::vec2> group_uvs;
    std::vector<glm::vec3> group_normals;

    printf("Indexing VBO total groups: %lu\n", mtl_vertices.size());
    for (int j=0; j<mtl_vertices.size(); j++ ) {
        group_vertices = mtl_vertices[j];
        if (use_uvs) group_uvs = mtl_uvs[j];
        if (use_normals) group_normals = mtl_normals[j];

        //printf("Group (%d/%d) size vertex %d\n", j, mtl_vertices.size(), group_vertices.size());
        for (int i=0; i<group_vertices.size(); i++ ) {
            //printf("\t subgroup size %d/%d %d/%d\n", j, mtl_vertices.size(), i, group_vertices.size());
            
            PackedVertex packed = {group_vertices[i], glm::vec2(), glm::vec3()};
            unsigned int index;
            bool found = getSimilarVertexIndex_fast( packed, VertexToOutIndex, index);
            glm::vec2 semantic_uv;
            semantic_uv.x = (float)j;
            semantic_uv.y = 0;

            found = false;
            if ( found ) {
                out_indices.push_back( index );
            } else {
                // Disable vertex indexing
                out_vertices.push_back( group_vertices[i]);
                out_semantics.push_back( semantic_uv);
                unsigned int newindex = (unsigned int)out_vertices.size() - 1;
                out_indices.push_back( newindex);
                if (use_uvs) out_uvs.push_back( group_uvs[i]);
                if (use_normals) out_normals.push_back( group_normals[i]);    
                VertexToOutIndex [ packed ] = newindex;
            }
        }
    }
}


// TODO: Inplement index_SEM from load MTL
void indexVBO_PLY(
    std::vector<std::vector<glm::vec3>> & mtl_vertices,
    std::vector<std::vector<glm::vec2>> & mtl_uvs,
    std::vector<std::vector<glm::vec3>> & mtl_normals,

    std::vector<unsigned int> & out_indices,
    std::vector<glm::vec3> & out_vertices,
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals,
    std::vector<glm::vec2> & out_semantics
) {
    std::map<PackedVertex,unsigned int> VertexToOutIndex;

    // In group j
    for (int j=0; j<mtl_vertices.size(); j++ ) {
        std::vector<glm::vec3> group_vertices = mtl_vertices[j];
        std::vector<glm::vec2> group_uvs = mtl_uvs[j];
        std::vector<glm::vec3> group_normals = mtl_normals[j];
        //float semantic_layer = (float) 0;
        //float semantic_layer = (float) j;

        uint group_size = group_vertices.size();
        if (group_uvs.size() < group_size)
            group_size = group_uvs.size();

        for (int i=0; i<group_size; i++ ) {
            PackedVertex packed = {group_vertices[i], group_uvs[i], group_normals[i]};
            unsigned int index;
            bool found = getSimilarVertexIndex_fast( packed, VertexToOutIndex, index);
            glm::vec2 semantic_uv;
            semantic_uv.x = (float)j;//semantic_layer;
            semantic_uv.y = 0;

            // Disable vertex indexing
            found = false;
            if ( found ) {
                out_indices.push_back( index );
            } else {
                out_vertices .push_back( group_vertices[i]);
                out_uvs      .push_back( group_uvs[i]);
                out_normals  .push_back( group_normals[i]);
                out_semantics.push_back( semantic_uv);
                unsigned int newindex = (unsigned int)out_vertices.size() - 1;
                out_indices  .push_back( newindex);

                VertexToOutIndex [ packed ] = newindex;
            }
        }
    }
}


void indexVBO_TBN(
    std::vector<glm::vec3> & in_vertices,
    std::vector<glm::vec2> & in_uvs,
    std::vector<glm::vec3> & in_normals,
    std::vector<glm::vec3> & in_tangents,
    std::vector<glm::vec3> & in_bitangents,

    std::vector<unsigned int> & out_indices,
    std::vector<glm::vec3> & out_vertices,
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals,
    std::vector<glm::vec3> & out_tangents,
    std::vector<glm::vec3> & out_bitangents
){
    // For each input vertex
    for ( unsigned int i=0; i<in_vertices.size(); i++ ){

        // Try to find a similar vertex in out_XXXX
        unsigned int index;
        bool found = getSimilarVertexIndex(in_vertices[i], in_uvs[i], in_normals[i],     out_vertices, out_uvs, out_normals, index);

        if ( found ){ // A similar vertex is already in the VBO, use it instead !
            out_indices.push_back( index );

            // Average the tangents and the bitangents
            out_tangents[index] += in_tangents[i];
            out_bitangents[index] += in_bitangents[i];
        }else{ // If not, it needs to be added in the output data.
            out_vertices.push_back( in_vertices[i]);
            out_uvs     .push_back( in_uvs[i]);
            out_normals .push_back( in_normals[i]);
            out_tangents .push_back( in_tangents[i]);
            out_bitangents .push_back( in_bitangents[i]);
            out_indices .push_back( (unsigned int)out_vertices.size() - 1 );
        }
    }
}

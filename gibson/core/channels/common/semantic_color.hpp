#ifndef SEMANTIC_COLOR_HPP
#define SEMANTIC_COLOR_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cstdlib>  //rand
#include <GL/glut.h>
#include <fstream>
#include <GLFW/glfw3.h>
#include "common/MTLtexture.hpp"
#include <algorithm>

std::map<std::string, std::vector<unsigned char>> color_map_2d3ds = {
    { "<UNK>", {0, 0, 0} },
    { "ceiling", {241, 255, 82} },
    { "floor", {102, 168, 226} },
    { "window", {0, 255, 0} }, 
    { "door", {113, 143, 65} }, 
    { "column", {89, 173, 163} },  
    { "beam", {254, 158, 137} },  
    { "wall", {190, 123, 75} }, 
    { "sofa", {100, 22, 116} },  
    { "chair", {0, 18, 141} }, 
    { "table", {84, 84, 84} }, 
    { "board", {85, 116, 127} }, 
    { "bookcase", {255, 31, 33} }, 
    { "clutter", {228, 228, 228} }, 
};


void color_coding_RAND (GLubyte color[]) {
    color[0] = (unsigned char) (rand()%255);
    color[1] = (unsigned char) (rand()%255);
    color[2] = (unsigned char) (rand()%255);
};


void color_coding_2D3DS (GLubyte color[], unsigned int id) {
    /* Parse a 24-bit integer as a RGB color. I.e. Convert to base 256
    Adapted from alexsax/2D-3D-Semantics 
    Args:
        index: An int. The first 24 bits will be interpreted as a color.
            Negative values will not work properly.
    Returns:
        color: A color s.t. get_index( get_color( i ) ) = i
    */
    unsigned char b = ( id ) % 256;          // least significant byte
    unsigned char g = ( id >> 8 ) % 256;
    unsigned char r = ( id >> 16 ) % 256;    // most significant byte 
    color[0] = r;
    color[1] = g;
    color[2] = b;
};


void color_coding_MP3D (GLubyte color[], unsigned int id) {
    unsigned char b = ( id ) % 256;          // least significant byte
    unsigned char g = ( id >> 8 ) % 256;
    unsigned char r = ( id >> 16 ) % 256;    // most significant byte 
    color[0] = r;
    color[1] = g;
    color[2] = b;
};


void color_coding_2D3DS_pretty (GLubyte color[], std::string name) {
    /* Color code 2D3DS semantics in a nice pretty way
     */
    std::string name_prefix = name.substr(0, name.find("_", 0));
    std::vector<unsigned char> assign_color = color_map_2d3ds.find(name_prefix)->second;
    color[0] = assign_color[0];
    color[1] = assign_color[1];
    color[2] = assign_color[2];
};

#endif
#ifndef SEMANTIC_COLOR_HPP
#define SEMANTIC_COLOR_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cstdlib>  //rand
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glx.h>


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

#endif
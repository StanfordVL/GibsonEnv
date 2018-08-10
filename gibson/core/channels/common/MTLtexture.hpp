#ifndef MTLTEXTURE_HPP
#define MTLTEXTURE_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <GL/glut.h>
#include <GL/glx.h>


struct Vector3
    {
        // Default Constructor
        Vector3()
        {
            X = 0.0f;
            Y = 0.0f;
            Z = 0.0f;
        }
        // Variable Set Constructor
        Vector3(float X_, float Y_, float Z_)
        {
            X = X_;
            Y = Y_;
            Z = Z_;
        }
        // Bool Equals Operator Overload
        bool operator==(const Vector3& other) const
        {
            return (this->X == other.X && this->Y == other.Y && this->Z == other.Z);
        }
        // Bool Not Equals Operator Overload
        bool operator!=(const Vector3& other) const
        {
            return !(this->X == other.X && this->Y == other.Y && this->Z == other.Z);
        }
        // Addition Operator Overload
        Vector3 operator+(const Vector3& right) const
        {
            return Vector3(this->X + right.X, this->Y + right.Y, this->Z + right.Z);
        }
        // Subtraction Operator Overload
        Vector3 operator-(const Vector3& right) const
        {
            return Vector3(this->X - right.X, this->Y - right.Y, this->Z - right.Z);
        }
        // Float Multiplication Operator Overload
        Vector3 operator*(const float& other) const
        {
            return Vector3(this->X *other, this->Y * other, this->Z - other);
        }

        // Positional Variables
        float X;
        float Y;
        float Z;
};

struct Material
    {
        Material()
        {
            name;
            Ns = 0.0f;
            Ni = 0.0f;
            d = 0.0f;
            illum = 0;
        }

        std::string name;        // Material Name
        Vector3 Ka;              // Ambient Color
        Vector3 Kd;              // Diffuse Color
        Vector3 Ks;              // Specular Color
        Vector3 Ke;
        float Ns;                // Specular Exponent
        float Ni;                // Optical Density
        float d;                 // Dissolve
        int illum;               // Illumination
        std::string map_Ka;      // Ambient Texture Map
        std::string map_Kd;      // Diffuse Texture Map
        std::string map_Ks;      // Specular Texture Map
        std::string map_Ns;      // Specular Hightlight Map
        std::string map_d;       // Alpha Texture Map
        std::string map_bump;    // Bump Map
};

struct TextureObj
{
    TextureObj()
    {
        name;
        textureID;
    }
    std::string name;
    GLuint textureID;
};


namespace algorithm {
// Split a String into a string array at a given token
        inline void split(const std::string &in,
            std::vector<std::string> &out,
            std::string token)
        {
            out.clear();

            std::string temp;

            for (int i = 0; i < int(in.size()); i++)
            {
                std::string test = in.substr(i, token.size());

                if (test == token)
                {
                    if (!temp.empty())
                    {
                        out.push_back(temp);
                        temp.clear();
                        i += (int)token.size() - 1;
                    }
                    else
                    {
                        out.push_back("");
                    }
                }
                else if (i + token.size() >= in.size())
                {
                    temp += in.substr(i, token.size());
                    out.push_back(temp);
                    break;
                }
                else
                {
                    temp += in[i];
                }
            }
        }

        // Get tail of string after first token and possibly following spaces
        inline std::string tail(const std::string &in)
        {
            size_t token_start = in.find_first_not_of(" \t");
            size_t space_start = in.find_first_of(" \t", token_start);
            size_t tail_start = in.find_first_not_of(" \t", space_start);
            size_t tail_end = in.find_last_not_of(" \t");
            if (tail_start != std::string::npos && tail_end != std::string::npos)
            {
                return in.substr(tail_start, tail_end - tail_start + 1);
            }
            else if (tail_start != std::string::npos)
            {
                return in.substr(tail_start);
            }
            return "";
        }

        // Get first token of string
        inline std::string firstToken(const std::string &in)
        {
            if (!in.empty())
            {
                size_t token_start = in.find_first_not_of(" \t");
                size_t token_end = in.find_first_of(" \t", token_start);
                if (token_start != std::string::npos && token_end != std::string::npos)
                {
                    return in.substr(token_start, token_end - token_start);
                }
                else if (token_start != std::string::npos)
                {
                    return in.substr(token_start);
                }
            }
            return "";
        }

        // Get element at given index position
        template <class T>
        inline const T & getElement(const std::vector<T> &elements, std::string &index)
        {
            int idx = std::stoi(index);
            if (idx < 0)
                idx = int(elements.size()) + idx;
            else
                idx--;
            return elements[idx];
        }
}

/* parse MTL file */
bool parseMTL(std::string mtlpath, std::vector<Material> & out_material);

/* Method to load an image into a texture using the freeimageplus library. */
/* Returns the texture ID or dies trying */
/* code from: https://r3dux.org/2014/10/how-to-load-an-opengl-texture-using-the-freeimage-library-or-freeimageplus-technically/ */
//GLuint loadTextureImages(std::string texturePathString, int Vrtx, GLenum minificationFilter = GL_LINEAR, GLenum magnificationFilter = GL_LINEAR);

/* Generate images of solid color and assign them to textures */
/* For mtl files with no associated maps (textureimages) */
//GLuint solidColorTexture(Vector3 Ka, int Vrtx, GLenum minificationFilter = GL_LINEAR, GLenum magnificationFilter = GL_LINEAR);

/* main function to parse MTL files, load or generate texture iamges and generate openGL texture IDs */
bool loadMTLtextures (std::string mtlpath, std::vector<TextureObj> & objtext, std::vector<std::string> OBJMaterial_name);


bool loadPLYtextures(std::vector<TextureObj> & objText, 
    //std::vector<std::string> PLYmaterial_name,
    std::vector<int> PLYmaterial_id);

#endif

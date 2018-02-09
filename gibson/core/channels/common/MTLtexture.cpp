#include <fstream>
#include <FreeImage.h>
#include <GL/glew.h>
#include <glfw3.h>
#include "MTLtexture.hpp"


/* parse MTL file */
bool parseMTL(const std::string * mtlpath, std::vector<Material> & out_material){
    ifstream inFile;
    printf("Parsing %s file for material textures.\n", mtlpath);

    /* try to open the file */
    inFile.open(mtlpath);
    if (!inFile) {
        printf("%s could not be opened. Are you in the right directory ? Don't forget to read the FAQ !\n", mtlpath); getchar();
        return 0;
    }

    /* verify the type of file */
    char filecode[4];
    fread(filecode, 1, 4, fp);
    if (strncmp(filecode, "mtl ", 4) != 0) {
        fclose(fp);
        return 0;
    }

    /*read file line by line and extract materials and properties*/
    std::vector<Material> LoadedMaterials;
    // Go through each line looking for material variables
    Material tempMaterial;
    bool listening = false;
    std::string curline;
    while (std::getline(file, curline)) {
        // new material and material name
        if (algorithm::firstToken(curline) == "newmtl") {
            if (!listening) {
                listening = true;
                if (curline.size() > 7) {
                    tempMaterial.name = algorithm::tail(curline);
                }
                else {
                    tempMaterial.name = "none";
                }
            }
            else {
                // Generate the material
                // Push Back loaded Material
                out_material.push_back(tempMaterial);
                // Clear Loaded Material
                tempMaterial = Material();
                if (curline.size() > 7) {
                    tempMaterial.name = algorithm::tail(curline);
                }
                else {
                    tempMaterial.name = "none";
                }
            }
        }
        // Ambient Color
        if (algorithm::firstToken(curline) == "Ka") {
                std::vector<std::string> temp;
                algorithm::split(algorithm::tail(curline), temp, " ");
                if (temp.size() != 3) { continue; }
                tempMaterial.Ka.X = std::stof(temp[0]);
                tempMaterial.Ka.Y = std::stof(temp[1]);
                tempMaterial.Ka.Z = std::stof(temp[2]);
        }
        // Diffuse Color
        if (algorithm::firstToken(curline) == "Kd") {
            std::vector<std::string> temp;
            algorithm::split(algorithm::tail(curline), temp, " ");
            if (temp.size() != 3) { continue; }
            tempMaterial.Kd.X = std::stof(temp[0]);
            tempMaterial.Kd.Y = std::stof(temp[1]);
            tempMaterial.Kd.Z = std::stof(temp[2]);
        }
        // Specular Color
        if (algorithm::firstToken(curline) == "Ks") {
            std::vector<std::string> temp;
            algorithm::split(algorithm::tail(curline), temp, " ");
            if (temp.size() != 3) { continue; }
            tempMaterial.Ks.X = std::stof(temp[0]);
            tempMaterial.Ks.Y = std::stof(temp[1]);
            tempMaterial.Ks.Z = std::stof(temp[2]);
        }
        //
        if (algorithm::firstToken(curline) == "Ke") {
            std::vector<std::string> temp;
            algorithm::split(algorithm::tail(curline), temp, " ");
            if (temp.size() != 3) { continue; }
            tempMaterial.Ke.X = std::stof(temp[0]);
            tempMaterial.Ke.Y = std::stof(temp[1]);
            tempMaterial.Ke.Z = std::stof(temp[2]);
        }
        // Specular Exponent
        if (algorithm::firstToken(curline) == "Ns") {
            tempMaterial.Ns = std::stof(algorithm::tail(curline));
        }
        // Optical Density
        if (algorithm::firstToken(curline) == "Ni") {
            tempMaterial.Ni = std::stof(algorithm::tail(curline));
        }
        // Dissolve
        if (algorithm::firstToken(curline) == "d") {
            tempMaterial.d = std::stof(algorithm::tail(curline));
        }
        // Illumination
        if (algorithm::firstToken(curline) == "illum") {
            tempMaterial.illum = std::stoi(algorithm::tail(curline));
        }
        // Ambient Texture Map
        if (algorithm::firstToken(curline) == "map_Ka") {
            tempMaterial.map_Ka = algorithm::tail(curline);
        }
        // Diffuse Texture Map
        if (algorithm::firstToken(curline) == "map_Kd") {
            tempMaterial.map_Kd = algorithm::tail(curline);
        }
        // Specular Texture Map
        if (algorithm::firstToken(curline) == "map_Ks") {
            tempMaterial.map_Ks = algorithm::tail(curline);
        }
        // Specular Hightlight Map
        if (algorithm::firstToken(curline) == "map_Ns") {
            tempMaterial.map_Ns = algorithm::tail(curline);
        }
        // Alpha Texture Map
        if (algorithm::firstToken(curline) == "map_d") {
            tempMaterial.map_d = algorithm::tail(curline);
        }
        // Bump Map
        if (algorithm::firstToken(curline) == "map_Bump" || algorithm::firstToken(curline) == "map_bump" || algorithm::firstToken(curline) == "bump") {
            tempMaterial.map_bump = algorithm::tail(curline);
        }
    }
        
    // Deal with last material
    // Push Back loaded Material
    out_material.push_back(tempMaterial);
    
    // Test to see if anything was loaded
    // If not return false
    if (out_material.empty())
        printf("The %s file gave no materials back.\n", mtlpath); getchar();
        return 0;
    
    printf("Found %i materials.\n", out_material.length())
    return true;
}


/* Method to load an image into a texture using the freeimageplus library. */
/* Returns the texture ID or dies trying */
/* code from: https://r3dux.org/2014/10/how-to-load-an-opengl-texture-using-the-freeimage-library-or-freeimageplus-technically/ */
GLuint loadTextureImages(std::string texturePathString, GLenum minificationFilter = GL_LINEAR, GLenum magnificationFilter = GL_LINEAR)
{

    // Get the texturePath as a pointer to a const char array to play nice with FreeImage
    const char* texturePath = texturePathString.c_str();

    // Determine the format of the image.
    // Note: The second paramter ('size') is currently unused, and we should use 0 for it.
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(texturePath , 0);

    // Image not found? Abort! Without this section we get a 0 by 0 image with 0 bits-per-pixel but we don't abort, which
    // you might find preferable to dumping the user back to the desktop.
    if (format == -1)
    {
        printf("Could not find image: %s - Aborting.\n", texturePathString);
        return;
    }

    // Found image, but couldn't determine the file format? Try again...
    if (format == FIF_UNKNOWN)
    {
        printf("Couldn't determine file format - attempting to get from file extension...\n");
       
        // ...by getting the filetype from the texturePath extension (i.e. .PNG, .GIF etc.)
        // Note: This is slower and more error-prone that getting it from the file itself,
        // also, we can't use the 'U' (unicode) variant of this method as that's Windows only.
        format = FreeImage_GetFIFFromFilename(texturePath);
 
        // Check that the plugin has reading capabilities for this format (if it's FIF_UNKNOWN,
        // for example, then it won't have) - if we can't read the file, then we bail out =(
        if ( !FreeImage_FIFSupportsReading(format) )
        {
            printf("Detected image format cannot be read!\n");
            return;
        }
    }

    // If we're here we have a known image format, so load the image into a bitap
    FIBITMAP* bitmap = FreeImage_Load(format, filename);
 
    // How many bits-per-pixel is the source image?
    int bitsPerPixel =  FreeImage_GetBPP(bitmap);

 
    // Convert our image up to 32 bits (8 bits per channel, Red/Green/Blue/Alpha) -
    // but only if the image is not already 32 bits (i.e. 8 bits per channel).
    // Note: ConvertTo32Bits returns a CLONE of the image data - so if we
    // allocate this back to itself without using our bitmap32 intermediate
    // we will LEAK the original bitmap data, and valgrind will show things like this:
    //
    // LEAK SUMMARY:
    //  definitely lost: 24 bytes in 2 blocks
    //  indirectly lost: 1,024,874 bytes in 14 blocks    <--- Ouch.
    //
    // Using our intermediate and cleaning up the initial bitmap data we get:
    //
    // LEAK SUMMARY:
    //  definitely lost: 16 bytes in 1 blocks
    //  indirectly lost: 176 bytes in 4 blocks
    //
    // All above leaks (192 bytes) are caused by XGetDefault (in /usr/lib/libX11.so.6.3.0) - we have no control over this.
    //
    FIBITMAP* bitmap32;
    if (bitsPerPixel == 32)
    {
        printf("Source image has %i bits per pixel. Skipping conversion.\n", bitsPerPixel);
        bitmap32 = bitmap;
    }
    else
    {
        printf("Source image has %i bits per pixel. Converting to 32-bit color.\n", bitsPerPixel);
        bitmap32 = FreeImage_ConvertTo32Bits(bitmap);
    }
 
    // Some basic image info - strip it out if you don't care
    int imageWidth  = FreeImage_GetWidth(bitmap32);
    int imageHeight = FreeImage_GetHeight(bitmap32);
    printf("Image: %s is of size: %i x %i.\n", texturePathString, imageWidth, imageHeight);

    // Get a pointer to the texture data as an array of unsigned bytes.
    // Note: At this point bitmap32 ALWAYS holds a 32-bit colour version of our image - so we get our data from that.
    // Also, we don't need to delete or delete[] this textureData because it's not on the heap (so attempting to do
    // so will cause a crash) - just let it go out of scope and the memory will be returned to the stack.
    GLubyte* textureData = FreeImage_GetBits(bitmap32);
 
    // Generate a texture ID and bind to it
    GLuint tempTextureID;
    glGenTextures(1, &tempTextureID);
    glBindTexture(GL_TEXTURE_2D, tempTextureID);

    // Construct the texture.
    // Note: The 'Data format' is the format of the image data as provided by the image library. FreeImage decodes images into
    // BGR/BGRA format, but we want to work with it in the more common RGBA format, so we specify the 'Internal format' as such.
    glTexImage2D(GL_TEXTURE_2D,    // Type of texture
                 0,                // Mipmap level (0 being the top level i.e. full size)
                 GL_RGBA,          // Internal format
                 imageWidth,       // Width of the texture
                 imageHeight,      // Height of the texture,
                 0,                // Border in pixels
                 GL_BGRA,          // Data format
                 GL_UNSIGNED_BYTE, // Type of texture data
                 textureData);     // The image data to use for this texture

    // Specify our minification and magnification filters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minificationFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magnificationFilter);

     // If we're using MipMaps, then we'll generate them here.
    // Note: The glGenerateMipmap call requires OpenGL 3.0 as a minimum.
    if (minificationFilter == GL_LINEAR_MIPMAP_LINEAR   ||
        minificationFilter == GL_LINEAR_MIPMAP_NEAREST  ||
        minificationFilter == GL_NEAREST_MIPMAP_LINEAR  ||
        minificationFilter == GL_NEAREST_MIPMAP_NEAREST)
    {
        glGenerateMipmap(GL_TEXTURE_2D);
    }
 
    // Check for OpenGL texture creation errors
    GLenum glError = glGetError();
    if(glError)
    {
        cout << "There was an error loading the texture: "<< filenameString << endl;
 
        switch (glError)
        {
            case GL_INVALID_ENUM:
                cout << "Invalid enum." << endl;
                break;
 
            case GL_INVALID_VALUE:
                cout << "Invalid value." << endl;
                break;
 
            case GL_INVALID_OPERATION:
                cout << "Invalid operation." << endl;
 
            default:
                cout << "Unrecognised GLenum." << endl;
                break;
        }
 
        cout << "See https://www.opengl.org/sdk/docs/man/html/glTexImage2D.xhtml for further details." << endl;
    }
 
    // Unload the 32-bit colour bitmap
    FreeImage_Unload(bitmap32);
 
    // If we had to do a conversion to 32-bit colour, then unload the original
    // non-32-bit-colour version of the image data too. Otherwise, bitmap32 and
    // bitmap point at the same data, and that data's already been free'd, so
    // don't attempt to free it again! (or we'll crash).
    if (bitsPerPixel != 32)
    {
        FreeImage_Unload(bitmap);
    }
 
    // Finally, return the texture ID
    return tempTextureID;
}


/* Generate images of solid color and assign them to textures */
/* For mtl files with no associated maps (textureimages) */
GLuint solidColorTexture(Vector3 Ka) {
    int imageWidth = 100;
    int imageHeight = 100;
    FIBITMAP* bitmap = FreeImage_Allocate(imageWidth, imageHeight, 32);
    RGBQuad color;

    color.rgbRed = Ka.X;
    color.rgbBlue = Ka.Y;
    color.rgbGreen = Ka.Z;

    if (!bitmap) {
        printf("You can't allocate the texture image.\n");
        return;
    }

    FreeImage.FillBackground(bitmap, color, FICO_DEFAULT);
    FIBITMAP* bitmap32;
    if (bitsPerPixel == 32)
    {
        printf("Source image has %i bits per pixel. Skipping conversion.\n", bitsPerPixel);
        bitmap32 = bitmap;
    }
    else
    {
        printf("Source image has %i bits per pixel. Converting to 32-bit color.\n", bitsPerPixel);
        bitmap32 = FreeImage_ConvertTo32Bits(bitmap);
    }

    GLubyte* textureData = FreeImage_GetBits(bitmap32);

    // Generate a texture ID and bind to it
    GLuint tempTextureID;
    glGenTextures(1, &tempTextureID);
    glBindTexture(GL_TEXTURE_2D, tempTextureID);

    // Construct the texture.
    // Note: The 'Data format' is the format of the image data as provided by the image library. FreeImage decodes images into
    // BGR/BGRA format, but we want to work with it in the more common RGBA format, so we specify the 'Internal format' as such.
    glTexImage2D(GL_TEXTURE_2D,    // Type of texture
                 0,                // Mipmap level (0 being the top level i.e. full size)
                 GL_RGBA,          // Internal format
                 imageWidth,       // Width of the texture
                 imageHeight,      // Height of the texture,
                 0,                // Border in pixels
                 GL_BGRA,          // Data format
                 GL_UNSIGNED_BYTE, // Type of texture data
                 textureData);     // The image data to use for this texture

    // Specify our minification and magnification filters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minificationFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magnificationFilter);
 
     // If we're using MipMaps, then we'll generate them here.
    // Note: The glGenerateMipmap call requires OpenGL 3.0 as a minimum.
    if (minificationFilter == GL_LINEAR_MIPMAP_LINEAR   ||
        minificationFilter == GL_LINEAR_MIPMAP_NEAREST  ||
        minificationFilter == GL_NEAREST_MIPMAP_LINEAR  ||
        minificationFilter == GL_NEAREST_MIPMAP_NEAREST)
    {
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    return tempTextureID;
}

/* main function to parse MTL files, load or generate texture iamges and generate openGL texture IDs */
std::vector<TextureObj> loadMTLtextures (const std::string * mtlpath) {
    std::vector<Material> parsed_mtl_file;
    if (parseMTL(mtlpath, parsed_mtl_file) == 0){
        printf("The Material list is empty! Nothing more to process.\n"); getchar();
        return 0;
    }

    std::vector<TextureObj> textureObjs;
    //  ----- Initialise the FreeImage library -----
    // Note: Flag is whether we should load ONLY local (built-in) libraries, so
    // false means 'no, use external libraries also', and 'true' means - use built
    // in libs only, so it's like using the library as a static version of itself.
    FreeImage_Initialise(true);
    
    // load texture images only if they exist, or assign standard color
    for ( unsigned int i = 0; i < parsed_mtl_file.length(); i++) {
        if (parsed_mtl_file[i].map_Ka != NULL|| parsed_mtl_file[i].map_Kd != NULL || parsed_mtl_file[i].map_Ks != NULL) {
            if (parsed_mtl_file[i].map_Ka != NULL) {
                std::string imagePath = parsed_mtl_file[i].map_Ka;
            }
            else if (parsed_mtl_file[i].map_Kd != NULL) {
                std::string imagePath = parsed_mtl_file[i].map_Kd;
            }
            else if (parsed_mtl_file[i].map_Ks != NULL) {
                std::string imagePath = parsed_mtl_file[i].map_Ks;
            }
            else {
                printf("Could not find any image path.\n");
                return;
            }
            std::string path = mtlpath;
            std::string texturePath;
            std::string rgbString = "rgb.mtl";
            std::string semanticString = "semantic.mtl";
            if (path.find(rgbString) != std::string::npos) {
                texturePath = path.erase(path.end()-rgbString.length(), path.end()) + imagePath;
            }
            else if (path.find(semanticString) != std::string::npos) {
                texturePath = path.erase(path.end()-semanticString.length(), path.end()) + imagePath;
            }
            else {
                printf("Could not find folder path to the image.\n");
                return;
            }

            TextureObj tempText;
            tempText.name = parsed_mtl_file[i].name;
            // Load an image and bind it to a texture
            tempText.textureID = = loadTexture(texturePath);
            textureObjs.pushback(tempText);
        }
        if (parsed_mtl_file[i].map_Ka == NULL|| parsed_mtl_file[i].map_Kd == NULL || parsed_mtl_file[i].map_Ks == NULL) {
            // Create a handle for our texture
            GLuint tempTextureID;
            TextureObj tempText;
            tempText.name = parsed_mtl_file[i].name;
            // Generate an image of solid texture and bind it to texture
            tempText.textureID = solidColorTexture(parsed_mtl_file[i].Ka);
            textureObjs.pushback(tempText);
        }
    }
    
    if (parsed_mtl_file.length() != textureObjs.length()) {
        printf("The number of parsed materials is not equal to the generated texture IDs (%i vs %i)", parsed_mtl_file.length(), textureIDs.length());
        return;
    }

    // todo: UV map coordinates howe to add it to the texture map/
    // maybe not here???

    return textureObjs;
}
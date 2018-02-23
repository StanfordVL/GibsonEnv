// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>  //rand
#include <X11/Xlib.h>
#include <chrono>
#include "boost/multi_array.hpp"
#include "boost/timer.hpp"

// Include GLEW
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glx.h>
#include "lodepng.h"

// Include GLFW
#include <glfw3.h>
GLFWwindow* window;

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

// Include cuda
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <cuda.h>

//using namespace glm;
using namespace std;

#include <common/shader.hpp>
#include <common/texture.hpp>
#include <common/controls.hpp>
#include <common/objloader.hpp>
#include <common/vboindexer.hpp>
#include "common/cmdline.h"
#include <common/render_cuda_f.h>

#include <common/MTLobjloader.hpp>
#include <common/MTLplyloader.hpp>
#include <common/MTLtexture.hpp>
#include <zmq.hpp>

#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define sleep(n)    Sleep(n)
#endif


// We would expect width and height to be 1024 and 768
int windowWidth = 256;
int windowHeight = 256;
size_t panoWidth = 2048;
size_t panoHeight = 1024;
int cudaDevice = -1;

//float camera_fov = 90.0f;
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
typedef Bool (*glXMakeContextCurrentARBProc)(Display*, GLXDrawable, GLXDrawable, GLXContext);
static glXCreateContextAttribsARBProc glXCreateContextAttribsARB = NULL;
static glXMakeContextCurrentARBProc   glXMakeContextCurrentARB   = NULL;

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


glm::vec3 GetOGLPos(int x, int y)
{
    GLint viewport[4];
    GLdouble modelview[16];
    GLdouble projection[16];
    GLfloat winX, winY, winZ;
    GLdouble posX, posY, posZ;

    glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
    glGetDoublev( GL_PROJECTION_MATRIX, projection );
    glGetIntegerv( GL_VIEWPORT, viewport );

    winX = (float)x;
    winY = (float)viewport[3] - (float)y;
    glReadPixels( x, int(winY), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ );

    gluUnProject( winX, winY, winZ, modelview, projection, viewport, &posX, &posY, &posZ);

    return glm::vec3(posX, posY, posZ);
}


bool save_screenshot(string filename, int w, int h, GLuint renderedTexture)
{
  // This prevents the images getting padded
  //when the width multiplied by 3 is not a multiple of 4
  glPixelStorei(GL_PACK_ALIGNMENT, 1);

  int nSize = w*h*3;
  // First let's create our buffer, 3 channels per Pixel
  //unsigned short* dataBuffer = (unsigned short*)malloc(nSize*sizeof(unsigned short));
  char* dataBuffer = (char*)malloc(nSize*sizeof(char));

  if (!dataBuffer) return false;

  // Let's fetch them from the backbuffer
  // We request the pixels in GL_BGR format, thanks to Berzeger for the tip
  glReadPixels((GLint)0, (GLint)0,
        (GLint)w, (GLint)h,
         GL_BGR, GL_UNSIGNED_SHORT, dataBuffer);

  unsigned short least = 65535;
  unsigned short most = 0;

  glGetTextureImage(renderedTexture, 0, GL_BLUE, GL_UNSIGNED_SHORT, nSize*sizeof(unsigned short), dataBuffer);


  // Convert little endian (default) to big endian
  for (int i = 0; i < nSize * 2 / 2; i++) {
      char* arr = (char*)dataBuffer;
      char tmp = arr[i * 2 + 1];
      arr[i * 2 + 1] = arr[i * 2];
      arr[i * 2] = tmp;
  }

  std::vector<unsigned char> png;

  unsigned error = lodepng::encode(filename, (unsigned char*)dataBuffer, w, h, LCT_RGB, 16);
  free(dataBuffer);

  return true;
}


void error_callback(int error, const char* description)
{
    cout << "Error callback" << endl;
    puts(description);
    printf("%X\n", error);
}

glm::mat4 str_to_mat(std::string str) {
    glm::mat4 mat = glm::mat4();
    std::string delimiter = " ";

    size_t pos = 0;
    size_t idx = 0;
    std::string token;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        token = str.substr(0, pos);
        mat[idx % 4][idx / 4] = std::stof(token);
        str.erase(0, pos + delimiter.length());
        idx += 1;
    }
    mat[idx % 4][idx / 4] = std::stof(str);

    return mat;
}


std::vector<size_t> str_to_vec(std::string str) {
    std::string delimiter = " ";
    std::vector<size_t> longs;
    size_t pos = 0;
    size_t idx = 0;
    std::string token;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        token = str.substr(0, pos);
        longs.push_back(std::stoul(token));
        str.erase(0, pos + delimiter.length());
        idx += 1;
    }
    longs.push_back(std::stoul(str));
    return longs;
}

void debug_mat(glm::mat4 mat, std::string name) {
    std::cout << "Debugging matrix " << name << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << mat[0][i] << " " << mat[1][i] << " " << mat[2][i] << " " << mat[3][i] << " " << std::endl;
    }
}


void render_processing() {

}


int main( int argc, char * argv[] )
{

    cmdline::parser cmdp;
    cmdp.add<std::string>("modelpath", 'd', "data model directory", true, "");
    cmdp.add<int>("GPU", 'g', "GPU index", false, 0);
    cmdp.add<int>("Width", 'w', "Render window width", false, 256);
    cmdp.add<int>("Height", 'h', "Render window height", false, 256);
    cmdp.add<int>("Smooth", 's', "Whether render depth only", false, 0);
    cmdp.add<int>("Normal", 'n', "Whether render surface normal", false, 0);
    cmdp.add<float>("fov", 'f', "field of view", false, 90.0);
    cmdp.add<int>("Semantic", 't', "Whether render semantics", false, 0);
    cmdp.add<int>("RGBfromMesh", 'm', "Whether render RGB directly from Mesh", false, 0);

    cmdp.parse_check(argc, argv);

    std::string model_path = cmdp.get<std::string>("modelpath");
    int GPU_NUM = cmdp.get<int>("GPU");
    int smooth = cmdp.get<int>("Smooth");
    int normal = cmdp.get<int>("Normal");
    int semantic = cmdp.get<int>("Semantic");
    int rgbMesh = cmdp.get<int>("RGBfromMesh");
    int ply;

    float camera_fov = cmdp.get<float>("fov");
    windowHeight = cmdp.get<int>("Height");
    windowWidth  = cmdp.get<int>("Width");

    std::string obj_path = model_path + "/modeldata/";
    std::string name_obj = obj_path + "out_res.obj";
    std::string name_ply = obj_path + "out_res.ply";

    if (smooth > 0) {
        name_obj = obj_path + "out_smoothed.obj";
        GPU_NUM = -1;
    }

    // if rendering normals
    if (normal > 0) {
        name_obj = obj_path + "rgb.obj";
        GPU_NUM = -2;
    }

    // if rendering semantics
    if (semantic > 0) {
        //name_obj = obj_path + "rgb.obj";
        name_obj = obj_path + "semantic.obj";
        ply = 1;
        GPU_NUM = -3;
    }

    // if rendering RGB from Mesh
    if (rgbMesh > 0) {
        name_obj = obj_path + "rgb.obj";
        GPU_NUM = -2;
    }

    std::string name_loc   = model_path + "/" + "sweep_locations.csv";


    glfwSetErrorCallback(error_callback);

    const char *displayName = NULL;
    Display* display = XOpenDisplay( displayName );

    if (display == NULL) {
        printf("Failed to properly open the display\n");
        return -1;
    }

    printf("opened this display, default %i\n", DefaultScreen(display));

    glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc) glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB" );
    glXMakeContextCurrentARB   = (glXMakeContextCurrentARBProc)   glXGetProcAddressARB( (const GLubyte *) "glXMakeContextCurrent"      );


    static int visualAttribs[] = { None };
    int numberOfFramebufferConfigurations = 0;

    //int err = glxewInit();


    //printf("starting from this point %d\n", err );
    GLXFBConfig* fbConfigs = glXChooseFBConfig( display, 0/*DefaultScreen(display)*/, visualAttribs, &numberOfFramebufferConfigurations );

    if (fbConfigs == NULL) {
        printf("Failed to properly set up frame buffer configurations\n");
        return -1;
    }

    int context_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
        GLX_CONTEXT_MINOR_VERSION_ARB, 2,
        GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_DEBUG_BIT_ARB,
        GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
        None
    };


    // printf("Running up to this point %X\n", (char *)fbConfigs);

    // This breaks if DISPLAY is not set as 0
    GLXContext openGLContext = glXCreateContextAttribsARB( display, fbConfigs[0], 0, True, context_attribs);


    // Initialise GLFW
    int pbufferAttribs[] = {
        GLX_PBUFFER_WIDTH,  32,
        GLX_PBUFFER_HEIGHT, 32,
        None
    };
    GLXPbuffer pbuffer = glXCreatePbuffer( display, fbConfigs[0], pbufferAttribs );


    // clean up:
    XFree( fbConfigs );
    XSync( display, False );


    if ( !glXMakeContextCurrent( display, pbuffer, pbuffer, openGLContext ) )
    {
        printf("Something is wrong\n");
        return -1;
        // something went wrong
    }


    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        getchar();
        return -1;
    }

    /*
    glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    */

    // Open a window and create its OpenGL context


    //window = glfwCreateWindow( windowWidth, windowHeight, "Depth Rendering", NULL, NULL);
    /*
    if( window == NULL ){
        fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
        getchar();
        glfwTerminate();
        return -1;
    }
    */

    //glfwMakeContextCurrent(window);


    // But on MacOS X with a retina screen it'll be 1024*2 and 768*2, so we get the actual framebuffer size:
    //glfwGetFramebufferSize(window, &windowWidth, &windowHeight);

    // Initialize GLEW


    glewExperimental = true; // Needed for core profile
    GLenum err = glewInit();
    if ( err!= GLEW_OK) {
        printf("Glew error %d\n", err);
        fprintf(stderr, "Failed to initialize GLEW %s\n", glewGetErrorString(err));
        getchar();
        //glfwTerminate();
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    //glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    // Hide the mouse and enable unlimited mouvement
    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Set the mouse at the center of the screen
    //glfwPollEvents();
    //glfwSetCursorPos(window, windowWidth/2, windowHeight/2);


    // Dark blue background
    //glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);


    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);
    //glDepthRange(1.0f, 0.0f);


    // Cull triangles which normal is not towards the camera
    //glEnable(GL_CULL_FACE);

    glDisable(GL_CULL_FACE);

    GLuint VertexArrayID;     // VAO
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // Create and compile our GLSL program from the shaders
    GLuint programID;
    if (normal == 0 && semantic == 0) {
        programID = LoadShaders( "./StandardShadingRTT.vertexshader", "./MistShadingRTT.fragmentshader" );
    } else if (normal >0 && semantic == 0) {
        programID = LoadShaders( "./NormalShadingRTT.vertexshader", "./NormalShadingRTT.fragmentshader" );
    } else if (normal == 0 && semantic > 0) {
        programID = LoadShaders( "./SemanticsShadingRTT.vertexshader", "./SemanticsShadingRTT.fragmentshader" );
    } else {
        printf("NEED TO ADJUST THE SHADERS!");
        programID = LoadShaders( "./StandardShadingRTT.vertexshader", "./MistShadingRTT.fragmentshader" );
    }

    // Get a handle for our "MVP" uniform
    GLuint MatrixID = glGetUniformLocation(programID, "MVP");
    GLuint ViewMatrixID = glGetUniformLocation(programID, "V");
    GLuint ModelMatrixID = glGetUniformLocation(programID, "M");


    std::vector<std::vector<glm::vec3>> mtl_vertices;
    std::vector<std::vector<glm::vec2>> mtl_uvs;
    std::vector<std::vector<glm::vec3>> mtl_normals;
    std::vector<std::string> material_name;
    std::vector<int> material_id;
    std::string mtllib;

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> normals;
    std::vector<TextureObj> TextObj;
    unsigned int num_layers;

    GLuint gArrayTexture(0);
    if ( semantic > 0) {
        // Prevent clamping
        /*
        glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
        glClampColorARB(GL_CLAMP_READ_COLOR_ARB, GL_FALSE);
        glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
        */
        // Do Y
        // Read the .obj file
        glShadeModel(GL_FLAT);

        //bool res = loadPLYfile(name_obj)
        std::cout << "Loading ply file\n";
        bool res;
        int num_vertices;
        if (ply > 0) {
            //res = loadPLY(obj_path.c_str(), vertices, uvs, normals);
            res = loadPLY_MTL(obj_path.c_str(), mtl_vertices, mtl_uvs, mtl_normals, material_name, material_id, mtllib, num_vertices);
            printf("From ply loaded total of %d vertices\n", num_vertices);
        } else {
            res = loadOBJ_MTL(name_obj.c_str(), mtl_vertices, mtl_uvs, mtl_normals, material_name, mtllib);
        }
        //res = loadOBJ(name_obj.c_str(), vertices, uvs, normals);
        if (res == false) { printf("Was not able to load the semantic.obj file.\n"); exit(-1); }
        else { printf("Semantic.obj file was loaded with success.\n"); }

        // Load the textures
        std::string mtl_path = obj_path + mtllib;
        bool MTL_loaded;
        if (ply > 0) {
            mtl_path = obj_path;
            // TODO: load actual mtl file for ply json
            // MTL_loaded = true;
            MTL_loaded = loadPLYtextures(TextObj, material_name, material_id);
        } else {
            MTL_loaded = loadMTLtextures(mtl_path, TextObj, material_name);    
        }
        if (MTL_loaded == false) { printf("Was not able to load textures\n"); exit(-1); }
        else { printf("Texture file was loaded with success, total: %d\n", TextObj.size()); }

        num_layers = TextObj.size();
        //Generate an array texture
        glGenTextures( 1, &gArrayTexture );
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D_ARRAY, gArrayTexture);

        //Create storage for the texture. (100 layers of 1x1 texels)
        glTexStorage3D( GL_TEXTURE_2D_ARRAY,
                      1,                    //No mipmaps as textures are 1x1
                      GL_RGB8,              //Internal format
                      1, 1,                 //width,height
                      num_layers            //Number of layers
                    );

        int layer_count = 0;
        for( unsigned int i(0); i!=num_layers;++i)
        {
            //Choose a random color for the i-essim image
            GLubyte color[3] = {(unsigned char) (rand()%255),
                                (unsigned char) (rand()%255),
                                (unsigned char) (rand()%255)};
            
            //Specify i-essim image
            glTexSubImage3D( GL_TEXTURE_2D_ARRAY,
                             0,                     //Mipmap number
                             0,0,i,                 //xoffset, yoffset, zoffset
                             1,1,1,                 //width, height, depth
                             GL_RGB,                //format
                             GL_UNSIGNED_BYTE,      //type
                             color);                //pointer to data
        }

        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    } else {
        bool res = loadOBJ(name_obj.c_str(), vertices, uvs, normals);
    }

    // Get a handle for our "myTextureSampler" uniform
    GLuint TextureID  = glGetUniformLocation(programID, "myTextureSampler");

    // Read our .obj file
    // Note: use unsigned int because of too many indices
    std::vector<unsigned int> indices;
    std::vector<glm::vec3> indexed_vertices;
    std::vector<glm::vec2> indexed_uvs;
    std::vector<glm::vec3> indexed_normals;
    std::vector<glm::vec2> indexed_semantics;

    if (semantic > 0) {
        indexVBO_MTL(mtl_vertices, mtl_uvs, mtl_normals, indices, indexed_vertices, indexed_uvs, indexed_normals, indexed_semantics);
        std::cout << "Finished indexing vertices v " << indexed_vertices.size() << " uvs " << indexed_uvs.size() << " normals " << indexed_normals.size() << " semantics " << indexed_semantics.size() << std::endl;
        std::cout << "Semantics ";
        //for (unsigned int i = 250000; i < 260000; i++) printf("%u (%f)", i, indexed_semantics[i].x);
        std::cout << std::endl;
    } else {
        indexVBO(vertices, uvs, normals, indices, indexed_vertices, indexed_uvs, indexed_normals);
    }


    // Load it into a VBO
    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, indexed_vertices.size() * sizeof(glm::vec3), &indexed_vertices[0], GL_STATIC_DRAW);

    GLuint uvbuffer;
    glGenBuffers(1, &uvbuffer);
    if (! ply > 0 && ! semantic > 0) {
        glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
        if (indexed_uvs.size() > 0) glBufferData(GL_ARRAY_BUFFER, indexed_uvs.size() * sizeof(glm::vec2), &indexed_uvs[0], GL_STATIC_DRAW);    
    }
    
    GLuint normalbuffer;
    glGenBuffers(1, &normalbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
    if (indexed_normals.size() > 0) glBufferData(GL_ARRAY_BUFFER, indexed_normals.size() * sizeof(glm::vec3), &indexed_normals[0], GL_STATIC_DRAW);

    GLuint semanticlayerbuffer;
    if (semantic > 0) {
        glGenBuffers(1, &semanticlayerbuffer);
        glBindBuffer(GL_ARRAY_BUFFER, semanticlayerbuffer);
        glBufferData(GL_ARRAY_BUFFER, indexed_semantics.size() * sizeof(glm::vec2), &indexed_semantics[0], GL_STATIC_DRAW);
    }

    // Generate a buffer for the indices as well
    GLuint elementbuffer;
    glGenBuffers(1, &elementbuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    // Get a handle for our "LightPosition" uniform
    glUseProgram(programID);
    GLuint LightID = glGetUniformLocation(programID, "LightPosition_worldspace");


    // ---------------------------------------------
    // Render to Texture - specific code begins here
    // ---------------------------------------------

    // The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
    // ER: Duplicate this six times
    GLuint FramebufferName = 0;
    glGenFramebuffers(1, &FramebufferName);
    glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

    // The texture we're going to render to
    GLuint renderedTexture;

    glGenTextures(1, &renderedTexture);

    // "Bind" the newly created texture : all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, renderedTexture);

    // Give an empty image to OpenGL ( the last "0" means "empty" )
    if (semantic > 0) {
      glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA32F, windowWidth, windowHeight, 0,GL_BLUE, GL_FLOAT, 0);
    } else {
      glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA32F, windowWidth, windowHeight, 0,GL_BLUE, GL_FLOAT, 0);  
    }
    
    // Poor filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // The depth buffer
    GLuint depthrenderbuffer;
    glGenRenderbuffers(1, &depthrenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, windowWidth, windowHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

    // Set "renderedTexture" as our colour attachement #0
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);

    GLenum DrawBuffers[2] = {GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT};
    //GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(2, DrawBuffers); // "1" is the size of DrawBuffers
      
    // Always check that our framebuffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      printf("Failed to properly draw buffers. Check your openGL frame buffer settings\n");
      return false;
    }


    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REP);
    std::cout << "GPU NUM:" << GPU_NUM  << " bound to port " << GPU_NUM + 5555 << std::endl;
    socket.bind ("tcp://127.0.0.1:"  + std::to_string(GPU_NUM + 5555));
    cudaGetDevice( &cudaDevice );
    int g_cuda_device = 0;
    if (GPU_NUM > 0) {
        cudaDevice = GPU_NUM;
    } else {
        cudaDevice = 0;
    }
    cudaSetDevice(cudaDevice);
    cudaGLSetGLDevice(cudaDevice);
    cudaGraphicsResource* resource;
    checkCudaErrors(cudaGraphicsGLRegisterImage(&resource, renderedTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
    
    std::cout << "CUDA DEVICE:" << cudaDevice << std::endl;
    int pose_idx = 0;
    zmq::message_t request;

    std::vector<unsigned int> cubeMapCoordToPanoCoord;
    for(size_t ycoord = 0; ycoord < panoHeight; ycoord++){
        for(size_t xcoord = 0; xcoord < panoWidth; xcoord++){
            size_t ind = 0;   //reordering[ycoord][xcoord][0];
            size_t corrx = 0; //reordering[ycoord][xcoord][1];
            size_t corry = 0; //reordering[ycoord][xcoord][2];

            cubeMapCoordToPanoCoord.push_back(
                ind * windowWidth * windowHeight +
                (windowHeight - 1 - corry) * windowWidth +
                corrx);
        }
    }

    unsigned int *d_cubeMapCoordToPanoCoord = copyToGPU(&(cubeMapCoordToPanoCoord[0]), cubeMapCoordToPanoCoord.size());

    float *cubeMapGpuBuffer = allocateBufferOnGPU(windowHeight * windowWidth * 6);
    cudaMemset(cubeMapGpuBuffer, 0, windowHeight * windowWidth * 6 * sizeof(float));

    do{

        //  Wait for next request from client
        socket.recv (&request);
        
        boost::timer t;

        std::string request_str = std::string(static_cast<char*>(request.data()), request.size());
        glm::mat4 viewMat = str_to_mat(request_str);

        // Measure speed
        glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
        glViewport(0,0,windowWidth,windowHeight); // Render on the whole framebuffer, complete from the lower left corner to the upper right

        int nSize = windowWidth*windowHeight*3*6;
        int nByte = nSize*sizeof(float);
        
        // --------------------------------------------------------------
        // ---------- RENDERING IN PANORAMA MODE ------------------------
        // ---------- Render to our framebuffer -------------------------
        // --------------------------------------------------------------
        
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use our shader
        glUseProgram(programID);

        // Compute the MVP matrix from keyboard and mouse input
        float fov = glm::radians(camera_fov);
        glm::mat4 ProjectionMatrix = glm::perspective(fov, 1.0f, 0.1f, 5000.0f); // near & far are not verified, but accuracy seems to work well
        glm::mat4 ViewMatrix =  getView(viewMat, 2);
        glm::mat4 viewMatPose = glm::inverse(ViewMatrix);

        glm::mat4 ModelMatrix = glm::mat4(1.0);

        glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

        // Send our transformation to the currently bound shader,
        // in the "MVP" uniform
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
        glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);
        glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &ViewMatrix[0][0]);

        glm::vec3 lightPos = glm::vec3(4,4,4);
        glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);

        // Bind our texture in Texture Unit 0
        glActiveTexture(GL_TEXTURE0);
        if (semantic > 0) {
            glBindTexture(GL_TEXTURE_2D_ARRAY, gArrayTexture);
            // Uniform variable: max_layer
            glUniform1i(num_layers, 3);
        }
        // Set our "myTextureSampler" sampler to use Texture Unit 0
        glUniform1i(TextureID, 0);


        std::cout << "Loop: binding buffers" << std::endl;
        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(
            0,                  // attribute
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
        );

        // 2nd attribute buffer : UVs
        if (! ply > 0 && ! semantic > 0) {
            glEnableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
            glVertexAttribPointer(
                1,                                // attribute
                2,                                // size
                GL_FLOAT,                         // type
                GL_FALSE,                         // normalized?
                0,                                // stride
                (void*)0                          // array buffer offset
            );
        }

        // 3rd attribute buffer : normals
        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
        glVertexAttribPointer(
            2,                                // attribute
            3,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            0,                                // stride
            (void*)0                          // array buffer offset
        );

        if (semantic > 0) {
            // 3rd attribute buffer : semantics
            glEnableVertexAttribArray(3);
            glBindBuffer(GL_ARRAY_BUFFER, semanticlayerbuffer);
            glVertexAttribPointer(
                3,                                // attribute
                2,                                // size
                GL_FLOAT,                         // type
                GL_FALSE,                         // normalized?
                0,                                // stride
                (void*)0                          // array buffer offset
            );
        }

        // Index buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);

        // Draw the triangles !
        glDrawElements(
            GL_TRIANGLES,      // mode
            indices.size(),    // count
            GL_UNSIGNED_INT,   // type
            (void*)0           // element array buffer offset
        );

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);


        int message_sz;
        int dim;
        if (normal > 0) {
            dim = 3;
            message_sz = windowWidth*windowHeight*sizeof(float)*dim;
        } else if (semantic > 0) {
            dim = 3;
            message_sz = windowWidth*windowHeight*sizeof(unsigned int)*dim;
        } else {
            dim = 1;
            message_sz = windowWidth*windowHeight*sizeof(float)*dim;
        }

        zmq::message_t reply (message_sz);

        if (semantic > 0) {
          // For semantics, we need to confine reply data values to unsigned integer
          float textureReadout[windowWidth*windowHeight*dim];
          int float_message_sz = windowWidth*windowHeight*sizeof(float)*dim;
          glGetTextureImage(renderedTexture, 0, GL_RGB, GL_FLOAT, float_message_sz, textureReadout);
          unsigned int * reply_data_handle = (unsigned int*)reply.data();
          float tmp_float;
          int offset;
          int pixel_count = 0;
          for (int i = 0; i < windowHeight; i++) {
            for (int j = 0; j < windowWidth; j++) {
              for (int k = 0; k < dim; k++) {
                offset = k;
                tmp_float = textureReadout[offset + (i * windowWidth + j) * dim];  
                reply_data_handle[offset + ((windowHeight - 1 -i) * windowWidth + j) * dim] = static_cast<unsigned int>(tmp_float);
              }
              /*
              if (pixel_count % 10000 == 0) {
                printf("Image pixel unsigned int %u %u %u\n", reply_data_handle[0 + ((windowHeight - 1 -i) * windowWidth + j) * dim], reply_data_handle[1 + ((windowHeight - 1 -i) * windowWidth + j) * dim], reply_data_handle[2 + ((windowHeight - 1 -i) * windowWidth + j) * dim]);
                printf("Image pixel float %f %f %f\n", textureReadout[0 + (i * windowWidth + j) * dim], textureReadout[1 + (i * windowWidth + j) * dim], textureReadout[2 + (i * windowWidth + j) * dim]);
              }
              */
              pixel_count += 1;
            }
          }
        } else {
          float * reply_data_handle = (float*)reply.data();
          glGetTextureImage(renderedTexture, 0, GL_BLUE, GL_FLOAT, message_sz, reply_data_handle);
          float tmp_float;
          int offset;
          // Revert the image from upside-down
          for (int i = 0; i < windowHeight/2; i++) {
            for (int j = 0; j < windowWidth; j++) {
              for (int k = 0; k < dim; k++) {
                offset = k;
                float * reply_data_handle = (float*)reply.data();
                tmp_float = reply_data_handle[offset + (i * windowWidth + j) * dim];
                reply_data_handle[offset + (i * windowWidth + j) * dim] = reply_data_handle[offset + ((windowHeight - 1 -i) * windowWidth + j) * dim];
                reply_data_handle[offset + ((windowHeight - 1 -i) * windowWidth + j) * dim] = tmp_float;
              }
            }
          }
        }
        socket.send (reply);


        //bool pano = False;

        //if (pano)
        //{
          /* 
            // ==============================================================
            // ========== RENDERING IN PANORAMA MODE=========================
            // ========== CURRENTLY DISABLED ================================
            // ==============================================================
            for (int k = 0; k < 6; k ++ )
            {
                // Render to our framebuffer
                // Clear the screen
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                // Use our shader
                glUseProgram(programID);

                // Compute the MVP matrix from keyboard and mouse input
                float fov = glm::radians(camera_fov);
                glm::mat4 ProjectionMatrix = glm::perspective(fov, 1.0f, 0.1f, 5000.0f); // near & far are not verified, but accuracy seems to work well
                glm::mat4 ViewMatrix =  getView(viewMat, k);
                glm::mat4 viewMatPose = glm::inverse(ViewMatrix);
                glm::mat4 ModelMatrix = glm::mat4(1.0);

                pose_idx ++;

                glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

                // Send our transformation to the currently bound shader,
                // in the "MVP" uniform
                glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
                glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);
                glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &ViewMatrix[0][0]);

                glm::vec3 lightPos = glm::vec3(4,4,4);
                glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);

                // Bind our texture in Texture Unit 0
                glActiveTexture(GL_TEXTURE0);
                //glBindTexture(GL_TEXTURE_2D, Texture);
                glBindTexture(GL_TEXTURE_2D_ARRAY, gArrayTexture);
                // Set our "myTextureSampler" sampler to use Texture Unit 0
                glUniform1i(TextureID, 0);


                // 1rst attribute buffer : vertices
                glEnableVertexAttribArray(0);
                glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
                glVertexAttribPointer(
                    0,                  // attribute
                    3,                  // size
                    GL_FLOAT,           // type
                    GL_FALSE,           // normalized?
                    0,                  // stride
                    (void*)0            // array buffer offset
                );

                // 2nd attribute buffer : UVs
                glEnableVertexAttribArray(1);
                glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
                glVertexAttribPointer(
                    1,                                // attribute
                    2,                                // size
                    GL_FLOAT,                         // type
                    GL_FALSE,                         // normalized?
                    0,                                // stride
                    (void*)0                          // array buffer offset
                );

                // 3rd attribute buffer : normals
                glEnableVertexAttribArray(2);
                glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
                glVertexAttribPointer(
                    2,                                // attribute
                    3,                                // size
                    GL_FLOAT,                         // type
                    GL_FALSE,                         // normalized?
                    0,                                // stride
                    (void*)0                          // array buffer offset
                );

                // Index buffer
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);

                // Draw the triangles !
                glDrawElements(
                    GL_TRIANGLES,      // mode
                    indices.size(),    // count
                    GL_UNSIGNED_INT,   // type
                    (void*)0           // element array buffer offset
                );

                glDisableVertexAttribArray(0);
                glDisableVertexAttribArray(1);
                glDisableVertexAttribArray(2);



                // Map the OpenGL texture buffer to CUDA memory space
                checkCudaErrors(cudaGraphicsMapResources(1, &resource));
                cudaArray_t writeArray;
                checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&writeArray, resource, 0, 0));

                // Copy the blue channel of the texture to the appropriate part of the cubemap that CUDA will use
                fillBlue(cubeMapGpuBuffer, writeArray, windowWidth * windowHeight * k, windowWidth, windowHeight);

                // Unmap the OpenGL texture so that it can be rewritten
                checkCudaErrors(cudaGraphicsUnmapResources(1, &resource));


            }
            checkCudaErrors(cudaStreamSynchronize(0));
            zmq::message_t reply (panoWidth*panoHeight*sizeof(float));
            projectCubeMapToEquirectangular((float*)reply.data(), cubeMapGpuBuffer, d_cubeMapCoordToPanoCoord, cubeMapCoordToPanoCoord.size(), (size_t) nSize/3);

            socket.send (reply);
            */
        //}
        //else {
        //}
    } while (true);
    
    // Cleanup VBO and shader
    glDeleteBuffers(1, &vertexbuffer);
    if (!ply > 0 && ! semantic > 0) glDeleteBuffers(1, &uvbuffer);
    glDeleteBuffers(1, &normalbuffer);
    if (semantic > 0) glDeleteBuffers(1, &semanticlayerbuffer);
    glDeleteBuffers(1, &elementbuffer);
    glDeleteProgram(programID);
    
    glDeleteFramebuffers(1, &FramebufferName);
    glDeleteTextures(1, &renderedTexture);
    glDeleteRenderbuffers(1, &depthrenderbuffer);
    glDeleteTextures(1, &gArrayTexture);
    glDeleteVertexArrays(1, &VertexArrayID);

    return 0;
}

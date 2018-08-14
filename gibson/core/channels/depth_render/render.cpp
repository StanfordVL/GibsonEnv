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
// Include GLEW
//#include <GL/glew.h>
#include <GL/glut.h>
#include "lodepng.h"

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
#include <common/objloader.hpp>
#include <common/vboindexer.hpp>
#include <common/cmdline.h>
#include <common/render_cuda_f.h>
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
//typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
//typedef Bool (*glXMakeContextCurrentARBProc)(Display*, GLXDrawable, GLXDrawable, GLXContext);
//static glXCreateContextAttribsARBProc glXCreateContextAttribsARB = NULL;
//static glXMakeContextCurrentARBProc   glXMakeContextCurrentARB   = NULL;

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

struct EGLInternalData2 {
    bool m_isInitialized;

    int m_windowWidth;
    int m_windowHeight;
    int m_renderDevice;


    EGLBoolean success;
    EGLint num_configs;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;

    EGLInternalData2()
    : m_isInitialized(false),
    m_windowWidth(0),
    m_windowHeight(0),
     m_renderDevice(-1) {}
};

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
    cmdp.add<int>("Semantic Source", 'r', "Semantic data source", false, 1);
    cmdp.add<int>("Semantic Color", 'c', "Semantic rendering color scheme", false, 1);

    cmdp.add<int>("Port", 'p', "Local port to render the channel", true, 5556);
    cmdp.parse_check(argc, argv);

    std::string model_path = cmdp.get<std::string>("modelpath");
    int PORT    = cmdp.get<int>("Port");
    int smooth = cmdp.get<int>("Smooth");
    int normal = cmdp.get<int>("Normal");
    int semantic = cmdp.get<int>("Semantic");
    int semantic_src = cmdp.get<int>("Semantic Source");
    int semantic_clr = cmdp.get<int>("Semantic Color");
    int ply;
    int gpu_idx = cmdp.get<int>("GPU");

    float camera_fov = cmdp.get<float>("fov");
    windowHeight = cmdp.get<int>("Height");
    windowWidth  = cmdp.get<int>("Width");

    std::string name_obj = model_path + "/mesh.obj";
    //std::string name_obj = model_path + "/gibson_decimated_z_up.obj";
    if (smooth > 0) {
        name_obj = model_path + "/out_smoothed.obj";
    }

    // if rendering semantics
    /* Semantic data source
     *  0: random texture
     *  1: Stanford 2D3DS
     *  2: Matterport3D
     */
    if (semantic > 0) {
        name_obj = model_path + "/semantic.obj";
        if (semantic_src == 1) ply = 0;
        if (semantic_src == 2) ply = 1;
    }

    std::string name_loc   = model_path + "/" + "camera_poses.csv";

    EGLBoolean success;
    EGLint num_configs;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;

    int m_windowWidth;
    int m_windowHeight;
    int m_renderDevice;

    m_windowWidth = windowWidth;
    m_windowHeight = windowHeight;
    m_renderDevice = -1;

    EGLint egl_config_attribs[] = {EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_DEPTH_SIZE,
        8,
        EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_BIT,
        EGL_NONE};

    EGLint egl_pbuffer_attribs[] = {
        EGL_WIDTH, m_windowWidth, EGL_HEIGHT, m_windowHeight,
        EGL_NONE,
    };

    EGLInternalData2* m_data = new EGLInternalData2();

    // Load EGL functions
    int egl_version = gladLoaderLoadEGL(NULL);
    if(!egl_version) {
        fprintf(stderr, "failed to EGL with glad.\n");
        exit(EXIT_FAILURE);

    };

    // Query EGL Devices
    const int max_devices = 32;
    EGLDeviceEXT egl_devices[max_devices];
    EGLint num_devices = 0;
    EGLint egl_error = eglGetError();
    if (!eglQueryDevicesEXT(max_devices, egl_devices, &num_devices) ||
        egl_error != EGL_SUCCESS) {
        printf("eglQueryDevicesEXT Failed.\n");
        m_data->egl_display = EGL_NO_DISPLAY;
    }

    printf("number of devices found %d\n", num_devices);
    m_data->m_renderDevice = gpu_idx;

    // Set display
    EGLDisplay display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                                  egl_devices[m_data->m_renderDevice], NULL);
    if (eglGetError() == EGL_SUCCESS && display != EGL_NO_DISPLAY) {
        int major, minor;
        EGLBoolean initialized = eglInitialize(display, &major, &minor);
        if (eglGetError() == EGL_SUCCESS && initialized == EGL_TRUE) {
            m_data->egl_display = display;
        }
    }

    if (!eglInitialize(m_data->egl_display, NULL, NULL)) {
        fprintf(stderr, "Unable to initialize EGL\n");
        exit(EXIT_FAILURE);
    }

    egl_version = gladLoaderLoadEGL(m_data->egl_display);
    if (!egl_version) {
        fprintf(stderr, "Unable to reload EGL.\n");
        exit(EXIT_FAILURE);
    }
    printf("Loaded EGL %d.%d after reload.\n", GLAD_VERSION_MAJOR(egl_version),
           GLAD_VERSION_MINOR(egl_version));


    m_data->success = eglBindAPI(EGL_OPENGL_API);
    if (!m_data->success) {
        // TODO: Properly handle this error (requires change to default window
        // API to change return on all window types to bool).
        fprintf(stderr, "Failed to bind OpenGL API.\n");
        exit(EXIT_FAILURE);
    }

    m_data->success =
    eglChooseConfig(m_data->egl_display, egl_config_attribs,
                    &m_data->egl_config, 1, &m_data->num_configs);
    if (!m_data->success) {
        // TODO: Properly handle this error (requires change to default window
        // API to change return on all window types to bool).
        fprintf(stderr, "Failed to choose config (eglError: %d)\n", eglGetError());
        exit(EXIT_FAILURE);
    }
    if (m_data->num_configs != 1) {
        fprintf(stderr, "Didn't get exactly one config, but %d\n", m_data->num_configs);
        exit(EXIT_FAILURE);
    }

    m_data->egl_surface = eglCreatePbufferSurface(
                                                  m_data->egl_display, m_data->egl_config, egl_pbuffer_attribs);
    if (m_data->egl_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "Unable to create EGL surface (eglError: %d)\n", eglGetError());
        exit(EXIT_FAILURE);
    }


    m_data->egl_context = eglCreateContext(
                                           m_data->egl_display, m_data->egl_config, EGL_NO_CONTEXT, NULL);
    if (!m_data->egl_context) {
        fprintf(stderr, "Unable to create EGL context (eglError: %d)\n",eglGetError());
        exit(EXIT_FAILURE);
    }

    m_data->success =
        eglMakeCurrent(m_data->egl_display, m_data->egl_surface, m_data->egl_surface,
                   m_data->egl_context);
    if (!m_data->success) {
        fprintf(stderr, "Failed to make context current (eglError: %d)\n", eglGetError());
        exit(EXIT_FAILURE);
    }

    if (!gladLoadGL(eglGetProcAddress)) {
        fprintf(stderr, "failed to load GL with glad.\n");
        exit(EXIT_FAILURE);
    }

    const GLubyte* ven = glGetString(GL_VENDOR);
    printf("GL_VENDOR=%s\n", ven);

    const GLubyte* ren = glGetString(GL_RENDERER);
    printf("GL_RENDERER=%s\n", ren);
    const GLubyte* ver = glGetString(GL_VERSION);
    printf("GL_VERSION=%s\n", ver);
    const GLubyte* sl = glGetString(GL_SHADING_LANGUAGE_VERSION);
    printf("GL_SHADING_LANGUAGE_VERSION=%s\n", sl);

/*
    glewExperimental = true; // Needed for core profile
    GLenum err = glewInit();
    if ( err!= GLEW_OK) {
        printf("Glew error %d\n", err);
        fprintf(stderr, "Failed to initialize GLEW %s\n", glewGetErrorString(err));
        getchar();
        //glfwTerminate();
        return -1;
    }
*/

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


    printf("finish loading shaders\n");
    // Get a handle for our "MVP" uniform
    GLuint MatrixID = glGetUniformLocation(programID, "MVP");
    GLuint ViewMatrixID = glGetUniformLocation(programID, "V");
    GLuint ModelMatrixID = glGetUniformLocation(programID, "M");


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


    GLuint gArrayTexture(0);
    if ( semantic > 0) {
        /* initialize random seed: */
        srand (0);
        glShadeModel(GL_FLAT);

        bool res;
        int num_vertices;
        if (ply > 0) {
            res = loadPLY_MTL(model_path.c_str(), mtl_vertices, mtl_uvs, mtl_normals, mtl_sem_centers, material_id, mtllib, num_vertices);
            printf("From ply loaded total of %d vertices\n", num_vertices);
        } else {
            res = loadOBJ_MTL(name_obj.c_str(), mtl_vertices, mtl_uvs, mtl_normals, mtl_sem_centers, material_name, mtllib);
            printf("From ply loaded total of %d vertices\n", num_vertices);
        }
        //res = loadOBJ(name_obj.c_str(), vertices, uvs, normals);
        if (res == false) { printf("Was not able to load the semantic.obj file.\n"); exit(-1); }
        else { printf("Semantic.obj file was loaded with success.\n"); }

        // Load the textures
        std::string mtl_path = model_path + "/" + mtllib;
        bool MTL_loaded;
        if (ply > 0) {
            mtl_path = model_path;
            // TODO: load actual mtl file for ply json
            // MTL_loaded = true;
            MTL_loaded = loadPLYtextures(TextObj, material_id);
        } else {
            MTL_loaded = loadMTLtextures(mtl_path, TextObj, material_name);
        }
        if (MTL_loaded == false) { printf("Was not able to load textures\n"); exit(-1); }
        else { printf("Texture file was loaded with success, total: %lu\n", TextObj.size()); }

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
            GLubyte color[3];
            unsigned int id = (uint)TextObj[i].textureID;

            if (semantic_clr == 1)
                color_coding_RAND(color); // Instance-by-Instance Color Coding
            else if (semantic_clr == 2) {
                if (semantic_src == 1)      { color_coding_2D3DS(color, id);}   // Stanford 2D3DS
                else if (semantic_src == 2) { color_coding_MP3D(color, id );} // Matterport3D
                else {printf("Invalid code for semantic source.\n"); exit(-1); }
            }  else {
                if (semantic_src == 1)      { color_coding_2D3DS_pretty(color, material_name[i]);}
                else {printf("Invalid code for semantic source.\n"); exit(-1); }
            }

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



    //std::cout << "Indexed object" << std::endl;

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
    if (semantic > 0) { glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA32F, windowWidth, windowHeight, 0,GL_BLUE, GL_FLOAT, 0); }
    else { glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA32F, windowWidth, windowHeight, 0,GL_BLUE, GL_FLOAT, 0); }

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
    socket.bind ("tcp://127.0.0.1:"  + std::to_string(PORT));
    cudaGetDevice( &cudaDevice );
    int g_cuda_device = 0;
    //if (GPU_NUM > 0) {
    //    cudaDevice = GPU_NUM;
    //} else {
    cudaDevice = 0;
    //}
    cudaSetDevice(cudaDevice);
    cudaGLSetGLDevice(cudaDevice);
    cudaGraphicsResource* resource;
    //checkCudaErrors(cudaGraphicsGLRegisterImage(&resource, renderedTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

    // std::cout << "CUDA DEVICE:" << cudaDevice << std::endl;
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
        glm::mat4 ProjectionMatrix = glm::perspective(fov, 1.0f, 0.1f, 5000000.0f); // near & far are not verified, but accuracy seems to work well
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
          if (normal > 0) {
            glGetTextureImage(renderedTexture, 0, GL_RGB, GL_FLOAT, message_sz, reply_data_handle);
          } else { glGetTextureImage(renderedTexture, 0, GL_BLUE, GL_FLOAT, message_sz, reply_data_handle); }
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

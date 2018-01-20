// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <X11/Xlib.h>

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
using namespace glm;
using namespace std;

#include <common/shader.hpp>
#include <common/texture.hpp>
#include <common/controls.hpp>
#include <common/objloader.hpp>
#include <common/vboindexer.hpp>
#include "common/cmdline.h"

#include <zmq.hpp>

#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>

#define sleep(n)    Sleep(n)
#endif



// We would expect width and height to be 1024 and 768
int windowWidth = 512;
int windowHeight = 512;

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
typedef Bool (*glXMakeContextCurrentARBProc)(Display*, GLXDrawable, GLXDrawable, GLXContext);
static glXCreateContextAttribsARBProc glXCreateContextAttribsARB = NULL;
static glXMakeContextCurrentARBProc   glXMakeContextCurrentARB   = NULL;


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


int main( int argc, char * argv[] )
{

    //cmdline::parser cmdp;
    //cmdp.add<std::string>("obj", 'b', "obj file name", true, "");
    //cmdp.parse_check(argc, argv);
    //std::string name_obj = cmdp.get<std::string>("obj");


    const char *displayName = NULL;
    Display* display = XOpenDisplay( displayName );

    if (display == NULL) {
        printf("Failed to properly open the display\n");
        return -1;
    }

    printf("opened this display, default %i\n", DefaultScreen(display));

    glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc) glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB" );
    glXMakeContextCurrentARB   = (glXMakeContextCurrentARBProc)   glXGetProcAddressARB( (const GLubyte *) "glXMakeContextCurrent"      );


    std::string name_obj = "1CzjpjNF8qk_HIGH.obj";
    cout << name_obj << endl;

    // Initialise GLFW

    
    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        getchar();
        return -1;
    }

    
    glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    // Open a window and create its OpenGL context


    window = glfwCreateWindow( windowWidth, windowHeight, "Depth Rendering", NULL, NULL);

    if( window == NULL ){
        fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
        getchar();
        glfwTerminate();
        return -1;
    }


    glfwMakeContextCurrent(window);


    // But on MacOS X with a retina screen it'll be 1024*2 and 768*2, so we get the actual framebuffer size:
    glfwGetFramebufferSize(window, &windowWidth, &windowHeight);

    // Initialize GLEW


    glewExperimental = true; // Needed for core profile
    GLenum err = glewInit();
    if ( err!= GLEW_OK) {
        printf("Glew error %d\n", err);
        fprintf(stderr, "Failed to initialize GLEW %s\n", glewGetErrorString(err));
        getchar();
        glfwTerminate();
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    // Hide the mouse and enable unlimited mouvement
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    // Set the mouse at the center of the screen
    glfwPollEvents();
    glfwSetCursorPos(window, windowWidth/2, windowHeight/2);


    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);


    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

    // Cull triangles which normal is not towards the camera
    glEnable(GL_CULL_FACE);

    GLuint VertexArrayID;     // VAO
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // Create and compile our GLSL program from the shaders
    GLuint programID = LoadShaders( "StandardShadingRTT.vertexshader", "StandardShadingRTT.fragmentshader" );

    // Get a handle for our "MVP" uniform
    GLuint MatrixID = glGetUniformLocation(programID, "MVP");
    GLuint ViewMatrixID = glGetUniformLocation(programID, "V");
    GLuint ModelMatrixID = glGetUniformLocation(programID, "M");

    // Load the texture
    GLuint Texture = loadDDS("uvmap.DDS");

    // Get a handle for our "myTextureSampler" uniform
    GLuint TextureID  = glGetUniformLocation(programID, "myTextureSampler");

    // Read our .obj file
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> normals;
    bool res = loadOBJ(name_obj.c_str(), vertices, uvs, normals);

    // Note: use unsigned int because of too many indices
    std::vector<unsigned int> indices;
    std::vector<glm::vec3> indexed_vertices;
    std::vector<glm::vec2> indexed_uvs;
    std::vector<glm::vec3> indexed_normals;
    indexVBO(vertices, uvs, normals, indices, indexed_vertices, indexed_uvs, indexed_normals);

    // Load it into a VBO

    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, indexed_vertices.size() * sizeof(glm::vec3), &indexed_vertices[0], GL_STATIC_DRAW);

    GLuint uvbuffer;
    glGenBuffers(1, &uvbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
    glBufferData(GL_ARRAY_BUFFER, indexed_uvs.size() * sizeof(glm::vec2), &indexed_uvs[0], GL_STATIC_DRAW);

    GLuint normalbuffer;
    glGenBuffers(1, &normalbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
    glBufferData(GL_ARRAY_BUFFER, indexed_normals.size() * sizeof(glm::vec3), &indexed_normals[0], GL_STATIC_DRAW);

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
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB32F, windowWidth, windowHeight, 0,GL_RGB, GL_FLOAT, 0);

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

    //// Alternative : Depth texture. Slower, but you can sample it later in your shader
    // ER: Duplicate this six times
    GLuint depthTexture;
    glGenTextures(1, &depthTexture);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT16, windowWidth, windowHeight, 0,GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Set "renderedTexture" as our colour attachement #0
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);

    //// Depth texture alternative :
    // ER: Duplicate this six times
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTexture, 0);


    // Set the list of draw buffers.
    // GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    // Pipeline: makes sure that output from 1st pass goes to 2nd pass
    GLenum DrawBuffers[2] = {GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT};
    glDrawBuffers(2, DrawBuffers); // "1" is the size of DrawBuffers

    // Always check that our framebuffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        return false;


    // The fullscreen quad's FBO
    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
    };

    GLuint quad_vertexbuffer;
    glGenBuffers(1, &quad_vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

    // Create and compile our GLSL program from the shaders
    GLuint quad_programID = LoadShaders( "Passthrough.vertexshader", "WobblyTexture.fragmentshader" );
    GLuint texID = glGetUniformLocation(quad_programID, "renderedTexture");
    GLuint timeID = glGetUniformLocation(quad_programID, "time");

       //double lastTime = glfwGetTime();
    double lastTime = 0;
    int nbFrames = 0;
    bool screenshot = false;

    int i = 0;



    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REP);
    socket.bind ("tcp://*:5555");



    do{

        zmq::message_t request;

        std::cout << "Waiting for incoming task" << std::endl;

        //  Wait for next request from client
        socket.recv (&request);
        std::cout << "Received Hello " << request.data() << std::endl;

        //printf("%s\n", request.data());

        // Measure speed
        //double currentTime = glfwGetTime();

        double currentTime = 0;

        //printf("Running main render loop %f\n");
        nbFrames++;
        if ( currentTime - lastTime >= 1.0 ){ // If last prinf() was more than 1sec ago
            // printf and reset
            printf("%f ms/frame %d fps\n", 1000.0/double(nbFrames), nbFrames);
            nbFrames = 0;
            lastTime += 1.0;
        }

        // Render to our framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
        glViewport(0,0,windowWidth,windowHeight); // Render on the whole framebuffer, complete from the lower left corner to the upper right

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use our shader
        glUseProgram(programID);

        // Compute the MVP matrix from keyboard and mouse input
        char filename[50];
        bool do_screenshot = computeMatricesFromInputs();
        glm::mat4 ProjectionMatrix = getProjectionMatrix();
        glm::mat4 ViewMatrix = getViewMatrix();
        glm::mat4 ModelMatrix = glm::mat4(1.0);

        /*
        printf("Before ");
        for (int i = 0; i < 16; i++)
            printf("%f ", ProjectionMatrix[i / 4][i % 4]);

        printf("\n");
        //BuildPerspProjMat(ProjectionMatrix, 1.0489180166567196, 1.0, 0.0, 128.0);
        printf("After ");
        for (int i = 0; i < 16; i++)
            printf("%f ", ProjectionMatrix[i / 4][i % 4]);

        printf("\n");
        */



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
        glBindTexture(GL_TEXTURE_2D, Texture);
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


        // Render to the screen
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        // Render on the whole framebuffer, complete from the lower left corner to the upper right
        glViewport(0,0,windowWidth,windowHeight);

        // Clear the screen
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use our shader
        glUseProgram(quad_programID);

        // Bind our texture in Texture Unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, renderedTexture);
        //glBindTexture(GL_TEXTURE_2D, depthTexture);
        // Set our "renderedTexture" sampler to use Texture Unit 0
        glUniform1i(texID, 0);

        glUniform1f(timeID, (float)(glfwGetTime()*10.0f) );

        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
        glVertexAttribPointer(
            0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
        );

        // Draw the triangles !
        glDrawArrays(GL_TRIANGLES, 0, 6); // 2*3 indices starting at 0 -> 2 triangles

        glDisableVertexAttribArray(0);



        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
        i ++;


        int nSize = windowWidth*windowHeight*3;
        int nByte = nSize*sizeof(unsigned short);
        // First let's create our buffer, 3 channels per Pixel
        unsigned short* dataBuffer = (unsigned short*)malloc(nByte);
        //char* dataBuffer = (char*)malloc(nSize*sizeof(char));

        if (!dataBuffer) return false;

        // Let's fetch them from the backbuffer
        // We request the pixels in GL_BGR format, thanks to Berzeger for the tip
        glReadPixels((GLint)0, (GLint)0,
            (GLint)windowWidth, (GLint)windowHeight,
             GL_BGR, GL_UNSIGNED_SHORT, dataBuffer);

        glGetTextureImage(renderedTexture, 0, GL_RGB, GL_UNSIGNED_SHORT, nSize*sizeof(unsigned short), dataBuffer);



        zmq::message_t reply (nByte);
        memcpy (reply.data (), (unsigned char*)dataBuffer, nByte);
        socket.send (reply);

        free(dataBuffer);

    }
    // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0 );

    // Cleanup VBO and shader
    glDeleteBuffers(1, &vertexbuffer);
    glDeleteBuffers(1, &uvbuffer);
    glDeleteBuffers(1, &normalbuffer);
    glDeleteBuffers(1, &elementbuffer);
    glDeleteProgram(programID);
    glDeleteTextures(1, &Texture);

    glDeleteFramebuffers(1, &FramebufferName);
    glDeleteTextures(1, &renderedTexture);
    glDeleteRenderbuffers(1, &depthrenderbuffer);
    glDeleteBuffers(1, &quad_vertexbuffer);
    glDeleteVertexArrays(1, &VertexArrayID);


    // Close OpenGL window and terminate GLFW
    glfwTerminate();

    return 0;
}

void BuildPerspProjMat(glm::mat4 &m, float fov, float aspect,
    float znear, float zfar) {
  float xymax = znear * tan(fov/2);
  float ymin = -xymax;
  float xmin = -xymax;

  float width = xymax - xmin;
  float height = xymax - ymin;

  float depth = zfar - znear;
  float q = -(zfar + znear) / depth;
  float qn = -2 * (zfar * znear) / depth;

  float w = 2 * znear / (width + 0.000001);
  w = w / aspect;
  float h = 2 * znear / (height + 0.000001);

  //m[0][0]  = w;
  m[0][1]  = (float) 0.0;
  m[0][2]  = (float) 0.0;
  m[0][3]  = (float) 0.0;

  m[1][0]  = (float) 0.0;
  //m[1][1]  = h;
  m[1][2]  = (float) 0.0;
  m[1][3]  = (float) 0.0;

  m[2][0]  = (float) 0.0;
  m[2][1]  = (float) 0.0;
  m[2][2] = q;
  m[2][3] = (float) -1.0;

  m[3][0] = (float) 0.0;
  m[3][1] = (float) 0.0;
  //m[3][2] = qn;
  m[3][3] = (float) 0.0;
}

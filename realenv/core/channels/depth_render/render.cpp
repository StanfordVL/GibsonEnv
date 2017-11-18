// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <X11/Xlib.h>
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
float camera_fov = 120.0f;

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

  int strange_count = 0;

  for (int i = 0; i < nSize - 50; i++) {
  	if (dataBuffer[i] < least) least = dataBuffer[i];
  	if (dataBuffer[i] > most) most = dataBuffer[i];
 }

  //least = least * 5000 *  65536.0f / 128.0f;
  //most = most * 5000 * 65536.0f / 128.0f;

  cout << filename << " " << "read least input " << least << " most input " <<  most << " strange count " << strange_count << endl;

  //Now the file creation
  //FILE *filePtr = fopen(filename.c_str(), "wb");
  //if (!filePtr) return false;

   /*
  unsigned char TGAheader[12]={0,0,2,0,0,0,0,0,0,0,0,0};
  unsigned char header[6] = { w%256,w/256,
			       h%256,h/256,
			       24,0};
  // We write the headers
  fwrite(TGAheader,	sizeof(unsigned char),	12,	filePtr);
  fwrite(header,	sizeof(unsigned char),	6,	filePtr);
  // And finally our image data
  //fwrite(dataBuffer,	sizeof(GLushort),	nSize,	filePtr);
  fwrite(dataBuffer,	sizeof(unsigned short),	nSize,	filePtr);
  */
  //fclose(filePtr);

  // Convert little endian (default) to big endian
  for (int i = 0; i < nSize * 2 / 2; i++) {
  	char* arr = (char*)dataBuffer;
  	char tmp = arr[i * 2 + 1];
  	arr[i * 2 + 1] = arr[i * 2];
  	arr[i * 2] = tmp;
  }

  std::vector<unsigned char> png;

  unsigned error = lodepng::encode(filename, (unsigned char*)dataBuffer, w, h, LCT_RGB, 16);
  //if(!error) lodepng::save_file(png, filename.c_str());

  //lodepng::lodepng_encode24(unsigned char** out, size_t* outsize,
  //                      const unsigned char* image, unsigned w, unsigned h);

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

int main( int argc, char * argv[] )
{

    cmdline::parser cmdp;
    cmdp.add<std::string>("modelpath", 'd', "data model directory", true, "");
    cmdp.add<int>("GPU", 'g', "GPU index", false, 0);
    cmdp.add<int>("Width", 'w', "Render window width", false, 256);
    cmdp.add<int>("Height", 'h', "Render window height", false, 256);
    cmdp.add<int>("Smooth", 's', "Whether render depth only", false, 0);
    cmdp.add<int>("Normal", 'n', "Whether render surface normal", false, 0);

    cmdp.parse_check(argc, argv);

    std::string model_path = cmdp.get<std::string>("modelpath");
    int GPU_NUM = cmdp.get<int>("GPU");
    int smooth = cmdp.get<int>("Smooth");
    int normal = cmdp.get<int>("Normal");

    windowHeight = cmdp.get<int>("Height");
    windowWidth  = cmdp.get<int>("Width");

    std::string name_obj = model_path + "/" + "modeldata/out_res.obj";
    if (smooth > 0) {
    	name_obj = model_path + "/" + "modeldata/out_smoothed.obj";
    	GPU_NUM = -1;
    }

    if (normal > 0) {
    	name_obj = model_path + "/" + "modeldata/rgb.obj";
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
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);


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
	if (normal == 0) {
		programID = LoadShaders( "./StandardShadingRTT.vertexshader", "./MistShadingRTT.fragmentshader" );
	} else {
		programID = LoadShaders( "./StandardShadingRTT.vertexshader", "./NormalShadingRTT.fragmentshader" );
	}
	
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
	//std::vector<short unsigned int> short_indices;
	//bool res = loadAssImp(name_ply.c_str(), short_indices, vertices, uvs, normals);

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
	glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA32F, windowWidth, windowHeight, 0,GL_BLUE, GL_FLOAT, 0);

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
	GLuint quad_programID = LoadShaders( "./Passthrough.vertexshader", "./WobblyTexture.fragmentshader" );
	GLuint texID = glGetUniformLocation(quad_programID, "renderedTexture");
	GLuint timeID = glGetUniformLocation(quad_programID, "time");

   	//double lastTime = glfwGetTime();
	double lastTime = 0;
	int nbFrames = 0;
	bool screenshot = false;

	int i = 0;


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

	//socket.recv (&request);
	// std::string request_str = std::string(static_cast<char*>(request.data()), request.size());
	// std::vector<size_t> new_idxs = str_to_vec(request_str);
    //size_t img_size = request.size();
    //size_t img_size = 1024 * 2048 * 3;
	//typedef boost::multi_array<uint, 3> array_type;
	//const int ndims=3;
	//boost::multi_array<uint, 3> reordering{boost::extents[img_size / sizeof(uint)][1][1]};
	// boost::array<array_type::size_type,ndims> orig_dims = {{img_size / sizeof(uint),1,1}};
	//boost::array<array_type::index,ndims> dims = {{1024, 2048, 3}};

	// multi_array reordering(orig_dims);

	//for (int i = 0; i < (img_size / sizeof(uint)) ; i++) {
        //reordering[i][0][0] = ((uint*)request.data())[i];
	//	reordering[i][0][0] = 0;
	//}

	//reordering.reshape(dims);

	std::vector<uint> cubeMapCoordToPanoCoord;
	for(size_t ycoord = 0; ycoord < panoHeight; ycoord++){
		for(size_t xcoord = 0; xcoord < panoWidth; xcoord++){
			size_t ind = 0;//reordering[ycoord][xcoord][0];
			size_t corrx = 0;//reordering[ycoord][xcoord][1];
			size_t corry = 0;//reordering[ycoord][xcoord][2];

			cubeMapCoordToPanoCoord.push_back(
				ind * windowWidth * windowHeight +
				(windowHeight - 1 - corry) * windowWidth +
				corrx);
		}
	}

	uint *d_cubeMapCoordToPanoCoord = copyToGPU(&(cubeMapCoordToPanoCoord[0]), cubeMapCoordToPanoCoord.size());
	//zmq::message_t reply0 (sizeof(float));
	//socket.send(reply0);

	float *cubeMapGpuBuffer = allocateBufferOnGPU(windowHeight * windowWidth * 6);
    cudaMemset(cubeMapGpuBuffer, 0, windowHeight * windowWidth * 6 * sizeof(float));

	do{


		//std::cout << "Realenv Channel Renderer: waiting for pose" << std::endl;

        //  Wait for next request from client
        socket.recv (&request);
		boost::timer t;

        std::string request_str = std::string(static_cast<char*>(request.data()), request.size());

        glm::mat4 viewMat = str_to_mat(request_str);

		// Measure speed
		// double currentTime = glfwGetTime();
		double currentTime = 0;

		nbFrames++;
		if ( currentTime - lastTime >= 1.0 ){
			printf("%f ms/frame %d fps\n", 1000.0/double(nbFrames), nbFrames);
			nbFrames = 0;
			lastTime += 1.0;
		}

        glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
        glViewport(0,0,windowWidth,windowHeight); // Render on the whole framebuffer, complete from the lower left corner to the upper right

        int nSize = windowWidth*windowHeight*3*6;
        //int nByte = nSize*sizeof(unsigned short);
        int nByte = nSize*sizeof(float);

        // create buffer, 3 channels per Pixel
        //float* dataBuffer = (float*)malloc(nByte);
        // First let's create our buffer, 3 channels per Pixel
        //float* dataBuffer = (float*)malloc(nByte);
        //char* dataBuffer = (char*)malloc(nSize*sizeof(char));

        //float * dataBuffer_c = (float * ) malloc(windowWidth*windowHeight * sizeof(float));
        //if (!dataBuffer) return false;
        //if (!dataBuffer_c) return false;

        bool pano = False;

        if (pano)
        {


            for (int k = 0; k < 6; k ++ )
            {
                // Render to our framebuffer

                // Clear the screen
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                // Use our shader
                glUseProgram(programID);

                // Compute the MVP matrix from keyboard and mouse input
                //computeMatricesFromInputs();
                //computeMatricesFromFile(name_loc);
                float fov = glm::radians(camera_fov);
                glm::mat4 ProjectionMatrix = glm::perspective(fov, 1.0f, 0.1f, 5000.0f); // near & far are not verified, but accuracy seems to work well
                glm::mat4 ViewMatrix =  getView(viewMat, k);
                //glm::mat4 ViewMatrix = getViewMatrix();
                glm::mat4 viewMatPose = glm::inverse(ViewMatrix);
                // printf("View (pose) matrix for skybox %d\n", k);
                // for (int i = 0; i < 4; ++i) {
                // 	printf("\t %f %f %f %f\n", viewMatPose[0][i], viewMatPose[1][i], viewMatPose[2][i], viewMatPose[3][i]);
                // 	//printf("\t %f %f %f %f\n", ViewMatrix[0][i], ViewMatrix[1][i], ViewMatrix[2][i], ViewMatrix[3][i]);
                // }

                glm::mat4 ModelMatrix = glm::mat4(1.0);

                pose_idx ++;

                //glm::mat4 tempMat = getViewMatrix();
                //debug_mat(tempMat, "csv");

                // glm::mat4 revertZ = glm::mat4();
                // revertZ[2][2] = -1;
                // glm::quat rotateZ_N90 = glm::quat(glm::vec3(0.0f, 0.0f, glm::radians(-90.0f)));
                // glm::quat rotateX_90 = glm::quat(glm::vec3(glm::radians(-90.0f), 0.0f, 0.0f));

                //glm::mat4 MVP = ProjectionMatrix * ViewMatrix * revertZ * ModelMatrix;
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

                /*
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
                */

                /*
                if (false) {
                    char buffer[100];
                    //printf("before: %s\n", buffer);
                    sprintf(buffer, "/home/jerry/Pictures/%s_mist.png", filename);
                    //printf("after: %s\n", buffer);
                    //printf("file name is %s\n", filename);
                    //printf("saving screenshot to %s\n", buffer);
                    save_screenshot(buffer, windowWidth, windowHeight, renderedTexture);
                }
                */

                // Swap buffers
                //glfwSwapBuffers(window);
                //glfwPollEvents();


                // Let's fetch them from the backbuffer
                // We request the pixels in GL_BGR format, thanks to Berzeger for the tip

                //glReadPixels((GLint)0, (GLint)0,
                //    (GLint)windowWidth, (GLint)windowHeight,
                //     GL_BGR, GL_UNSIGNED_SHORT, dataBuffer);
                //glReadPixels((GLint)0, (GLint)0,
                //    (GLint)windowWidth, (GLint)windowHeight,
                //     GL_BGR, GL_FLOAT, dataBuffer);

                //glGetTextureImage(renderedTexture, 0, GL_RGB, GL_UNSIGNED_SHORT, nSize*sizeof(unsigned short), dataBuffer);
                // float* loc = dataBuffer + windowWidth*windowHeight * k;
                // glGetTextureImage(renderedTexture, 0, GL_BLUE, GL_FLOAT,
                    // (nSize/3)*sizeof(float), loc);

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

            //std::cout << "Render time: " << t.elapsed() << std::endl;
            socket.send (reply);

            //free(dataBuffer);
            //free(dataBuffer_c);
        }
        else {
        //Pinhole mode

            // Render to our framebuffer

                // Clear the screen
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                // Use our shader
                glUseProgram(programID);

                // Compute the MVP matrix from keyboard and mouse input
                //computeMatricesFromInputs();
                //computeMatricesFromFile(name_loc);
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

                zmq::message_t reply (windowWidth*windowHeight*sizeof(float));
                float * reply_data_handle = (float*)reply.data();
                glGetTextureImage(renderedTexture, 0, GL_BLUE, GL_FLOAT, windowWidth * windowHeight *sizeof(float), reply_data_handle);

                //std::cout << "Render time: " << t.elapsed() << std::endl;


                float tmp;

                for (int i = 0; i < windowHeight/2; i++)
                    for (int j = 0; j < windowWidth; j++) {
                    tmp = reply_data_handle[i * windowWidth + j];
                    reply_data_handle[i * windowWidth + j] = reply_data_handle[(windowHeight - 1 -i) * windowWidth + j];
                    reply_data_handle[(windowHeight - 1 -i) * windowWidth + j] = tmp;
                }
                socket.send (reply);

                //free(dataBuffer);
                //free(dataBuffer_c);

        }



	} while (true);
	// Check if the ESC key was pressed or the window was closed
	//while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
	//	   glfwWindowShouldClose(window) == 0 );

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
	//glfwTerminate();

	return 0;
}

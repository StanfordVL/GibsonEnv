// Include GLFW
#include <glfw3.h>
extern GLFWwindow* window; // The "extern" keyword here is to access the variable "window" declared in tutorialXXX.cpp. This is a hack to keep the tutorials simple. Please avoid this.

// Include GLM
#include <glm/glm.hpp>
//#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <stdio.h>
#include <cassert>
#include <cstring>

using namespace glm;

#include "controls.hpp"

glm::mat4 ViewMatrix;
glm::mat4 ProjectionMatrix;

glm::mat4 getViewMatrix(){
	return ViewMatrix;
}
glm::mat4 getProjectionMatrix(){
	return ProjectionMatrix;
}


// Initial position : on +Z
// point 0 view 1
//glm::vec3 position = glm::vec3( 0.033474, 1.53312, 0.002227 );
// Location: rotate by 90 along x
// original (blender): 0.033474, -0.002227, 1.53312
// point 0 view 2
//glm::vec3 position = glm::vec3( -1.096474, 1.535375, -0.124639 ); // point 0 view 2 adapted
// original (blender): -1.096474, 0.124639, 1.535375
glm::vec3 position = glm::vec3(-1.096474, 0.124639, 1.535375); // point 0 view 2 original (blender):

// Point 6 view 0
//glm::vec3 position = glm::vec3( -1.096474, 1.535375, -0.124639 );

//glm::rotateX(position, glm::radians(90.0f)); 
//glm::rotateX(position, glm::radians(90.0f));
// Initial horizontal angle : toward -Z
float horizontalAngle = 3.14f;
// Initial vertical angle : none
float verticalAngle = 0.0f;
// Initial Field of View
float initialFoV = 45.0f;

float speed = 3.0f; // 3 units / second
float mouseSpeed = 0.005f;

float currentPoseStartTime = 0;
int currentPoseRotCount = 0;

void getPositionRotation(glm::vec3 &position, float& rotX, float& rotY, float& rotZ, char* filename) {
	// Change position, rotation and Z value
	printf("Updating position rotation\n");
	const char * path = "/home/jerry/Desktop/view3d/realenv/depth/posefile";
	FILE * file = fopen(path, "r");
	if( file == NULL ){
		printf("Impossible to open the file !\n");
	}
	printf("Done reading pose file\n");
	int i = -1;
	float pos[3], rot[3];
	char namebuf[50];
	while (i < currentPoseRotCount) {
		//printf("current i: %d\n", i);
		//printf("original vec %f %f %f\n", position[0], position[1], position[2]);

		memset(namebuf, 0, 50);
		int count = fscanf(file, "%f %f %f %f %f %f %s\n", &pos[0], &pos[1], &pos[2], &rot[0], &rot[1], &rot[2], namebuf );
		// printf("current count: %d %s\n", count, namebuf);
		assert(count == 7);
		
		//fgets(filename, 30, file);
		//printf("current count: %d %s\n", count, filename);
		i ++;
	}
	position[0] = pos[0];
	position[1] = pos[1];
	position[2] = pos[2];
	rotX = rot[0];
	rotY = rot[1];
	rotZ = rot[2];
	strcpy(filename, namebuf);
	printf("Successfully read pose file line %d\n", currentPoseRotCount);
	fclose(file);
}


bool computeMatricesFromInputs(char* filename){

	bool do_screenshot = false;

	// glfwGetTime is called only once, the first time this function is called
	static double lastTime = glfwGetTime();

	// Compute time difference between current and last frame
	double currentTime = glfwGetTime();
	float deltaTime = float(currentTime - lastTime);

	// Get mouse position
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	// Reset mouse position for next frame
	glfwSetCursorPos(window, 512/2, 512/2);

	// Compute new orientation
	horizontalAngle += mouseSpeed * float( 512/2 - xpos );
	verticalAngle   += mouseSpeed * float( 512/2 - ypos );

	// Direction : Spherical coordinates to Cartesian coordinates conversion
	glm::vec3 direction(
		cos(verticalAngle) * sin(horizontalAngle), 
		sin(verticalAngle),
		cos(verticalAngle) * cos(horizontalAngle)
	);
	
	// Right vector
	glm::vec3 right = glm::vec3(
		sin(horizontalAngle - 3.14f/2.0f), 
		0,
		cos(horizontalAngle - 3.14f/2.0f)
	);
	
	// Up vector
	// glm::vec3 up = glm::cross( right, direction );
	glm::vec4 up4 = glm::vec4(0.0, 1.0, 0.0, 1.0);

	// Move forward
	if (glfwGetKey( window, GLFW_KEY_UP ) == GLFW_PRESS){
		position += direction * deltaTime * speed;
	}
	// Move backward
	if (glfwGetKey( window, GLFW_KEY_DOWN ) == GLFW_PRESS){
		position -= direction * deltaTime * speed;
	}
	// Strafe right
	if (glfwGetKey( window, GLFW_KEY_RIGHT ) == GLFW_PRESS){
		position += right * deltaTime * speed;
	}
	// Strafe left
	if (glfwGetKey( window, GLFW_KEY_LEFT ) == GLFW_PRESS){
		position -= right * deltaTime * speed;
	}

	float FoV = initialFoV;// - 5 * glfwGetMouseWheel(); // Now GLFW 3 requires setting up a callback for this. It's a bit too complicated for this beginner's tutorial, so it's disabled instead.

	// TODO: control here
	// Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	// ProjectionMatrix = glm::perspective(glm::radians(FoV), 4.0f / 3.0f, 0.1f, 100.0f);
	// BuildPerspProjMat(ProjectionMatrix, 1.0489180166567196, 1.0, 0.0, 128.0);
	
	//ProjectionMatrix = glm::perspective(1.0489180166567196f, 1.0f, 0.1f, 5000.0f);
	//ProjectionMatrix = glm::perspective(1.0489180166567196f, 1.0f, 0.0f, 128.0f);
	// Camera matrix
	// Point 0 view 1
	/*float rotationX = 1.2462860345840454;
	float rotationY = -0.009244712069630623;
	float rotationZ = -1.2957184314727783;
	*/
	// Point 0 view 2
	float rotationX = 1.3605239391326904;
	float rotationY = -0.009078502655029297;
	float rotationZ = -1.441698670387268;

	ProjectionMatrix = glm::perspective(0.9698680134771724f, 1.0f, 0.1f, 5000.0f);

	// Point 6 view 0
	//float rotationX = 1.4468656778335571;
	//float rotationY = -0.00613052025437355;
	//float rotationZ = -0.22472861409187317;


	if (currentTime - currentPoseStartTime > 1) {
		getPositionRotation(position, rotationX, rotationY, rotationZ, filename);
		currentPoseStartTime = currentTime;
		currentPoseRotCount += 1;
		do_screenshot = true;
	}

	//glm::vec3 pose_direction = glm::vec3( 0.856556 -0.241792 -0.4559042); 
	// blender  Vector((0.9129738807678223, 0.2546083927154541, -0.31883105635643005))
	// Blender initial camera direction: (0, 0, -1)
	glm::vec4 pose_d = glm::vec4(0.0, 0.0, -1.0, 1.0);
	
	glm::mat4 pose_trans = glm::mat4(1.0);

	pose_trans = glm::rotate(pose_trans, -rotationX, glm::vec3(1.0f, 0.0f, 0.0f));
	pose_trans = glm::rotate(pose_trans, -rotationY, glm::vec3(0.0f, 1.0f, 0.0f));
	pose_trans = glm::rotate(pose_trans, -rotationZ, glm::vec3(0.0f, 0.0f, 1.0f));

	pose_d = pose_trans * pose_d;
	up4    = pose_trans * up4;
	
	//pose_d[1] = - pose_d[1];
	//glm::mat4 pose_trans = glm::mat4(1.0);

	/*
	pose_trans = glm::rotate(pose_trans, rotationX, glm::vec3(1.0f, 0.0f, 0.0f));
	pose_d = pose_trans * pose_d;
	pose_trans = glm::mat4(1.0);
	pose_trans = glm::rotate(pose_trans, rotationY, glm::vec3(0.0f, 1.0f, 0.0f));
	pose_d = pose_trans * pose_d;
	pose_trans = glm::mat4(1.0);
	pose_trans = glm::rotate(pose_trans, rotationZ, glm::vec3(0.0f, 0.0f, 1.0f));
	pose_d = pose_trans * pose_d;

	*/
	// pose_d[1] = - pose_d[1];

	glm::vec3 pose_direction(pose_d);
	glm::vec3 up(up4);
	//printf("pose direction %f %f %f %f\n", pose_d[0], pose_d[1], pose_d[2], pose_d[3]);
	// printf("pose direction %f %f %f\n", pose_direction[0], pose_direction[1], pose_direction[2]);
	//printf("     direction %f %f %f\n", direction[0], direction[1], direction[2]);

	//direction = glm::vec3( 0.912973, -0.2546083, -0.3188310); 

	// pose_direction = glm::rotateX(pose_direction, rotationX);
	//pose_direction = glm::rotateY(pose_direction, rotationY);
	//pose_direction = glm::rotateZ(pose_direction, rotationZ);

	//glm::vec3 pose_direction = glm::vec3(-0.094399, 0.601198, -0.793505);

	// printf("moved direction %f %f %f\n", direction[0], direction[1], direction[2]);

	// printf("final direction %f %f %f\n", pose_direction[0], pose_direction[1], pose_direction[2]);

	// First way: lookAt function
	
	/*
	ViewMatrix       = glm::lookAt(
								position,           // Camera is here
								position+pose_direction, // and looks here : at the same position, plus "direction"
								up                  // Head is up (set to 0,-1,0 to look upside-down)
						   );


	printf("First   view matrix (no translate)\n");
	for (int i = 0; i < 4; ++i) {
		printf("\t %f %f %f %f\n", ViewMatrix[0][i], ViewMatrix[1][i], ViewMatrix[2][i], ViewMatrix[3][i]);
	}
	*/
	
	/*
	printf("Current view matrix\n");
	for (int i = 0; i < 4; ++i) {
		printf("\t %f %f %f %f\n", ViewMatrix[0][i], ViewMatrix[1][i], ViewMatrix[2][i], ViewMatrix[3][i]);
	}
	*/
	
	//printf("Up  vector: %f %f %f\n", up[0], up[1], up[2]);
	//printf("pos vector: %f %f %f\n", position[0], position[1], position[2]);
	
	// Third way
	glm::quat viewDirection;
	glm::vec3 viewDirectionEuler(rotationX, rotationY, rotationZ);
	viewDirection = glm::quat(viewDirectionEuler);

	ViewMatrix = glm::inverse(glm::translate(glm::mat4(1.0), position) * glm::toMat4(viewDirection));
	//ViewMatrix = glm::toMat4(-viewDirection) * glm::translate(glm::mat4(1.0), -position);
	
	/*printf("Third   view matrix\n");
	for (int i = 0; i < 4; ++i) {
		printf("\t %f %f %f %f\n", ViewMatrix[0][i], ViewMatrix[1][i], ViewMatrix[2][i], ViewMatrix[3][i]);
	}*/

	// For the next frame, the "last time" will be "now"
	lastTime = currentTime;

	return do_screenshot;
}
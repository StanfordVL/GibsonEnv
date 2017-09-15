#ifndef CONTROLS_HPP
#define CONTROLS_HPP

#include <string>

bool computeMatricesFromInputs();
bool computeMatricesFromFile(std::string filename);

glm::mat4 getViewMatrix();
glm::mat4 getProjectionMatrix();

#endif
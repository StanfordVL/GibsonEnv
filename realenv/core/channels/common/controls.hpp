#ifndef CONTROLS_HPP
#define CONTROLS_HPP

#include <string>

bool computeMatricesFromInputs();
bool computeMatricesFromFile(std::string filename);

glm::mat4 getViewMatrix();
glm::mat4 getProjectionMatrix();

glm::mat4 getView(glm::mat4 source, int k);

#endif
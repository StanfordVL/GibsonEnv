#include "plyloader.h"

bool loadPLYfile(std::string path, int * nelems, char *** elem_names) {
  const char * plyPath = path.c_str();
  FILE * pFile;
  pFile = fopen(plyPath, "r");
  printf("Loading the ply file (%s).\n", plyPath);
  PlyFile * loaded_PlyFile = ply_read(pFile, nelems, elem_names);
  fclose(pFile);

  return true;
}

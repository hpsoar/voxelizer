#include "voxelizer_demo.h"
#include <GL/glut.h>
#include "cheetah/core/options.h"

using namespace voxel;
int main(int argc, char *argv[]) {
  glutInit(&argc, argv);
  cheetah::core::InitOptions();
  
  VoxelizerDemo demo;
  int dim = 128;
  bool uniform = false;
  if (argc > 3 && atoi(argv[3]) != 0) uniform = true;
  if (argc > 2) dim = atoi(argv[2]); 
  if (argc > 1) demo.Initialize(argv[1], dim, uniform);
  else demo.Initialize("bunny.obj", dim, uniform);
  demo.Run();
  return 0;
}
#include "voxelizer_demo.h"
#include <GL/glut.h>
#include "axle/core/options.h"

using namespace voxel;
int main(int argc, char *argv[]) {
  glutInit(&argc, argv);
  ax::InitOptions();
  
  VoxelizerDemo demo;
  int dim = 128;
  bool uniform = true;
  if (argc > 3 && atoi(argv[3]) != 0) uniform = true;
  if (argc > 2) dim = atoi(argv[2]); 
  if (argc > 1) demo.Initialize(argv[1], dim, uniform);
  else demo.Initialize("mahua2.obj", dim, uniform);
  demo.Run();
  return 0;
}
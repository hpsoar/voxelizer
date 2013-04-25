#ifndef VOXELIZER_VOXELIZER_TEST_H
#define VOXELIZER_VOXELIZER_TEST_H

#include "axle/core/settings.h"

#include "axle/cg/fps_counter.h"
#include "axle/cg/camera.h"
#include <GL/glew.h>
#include "axle/ui/glut_window.h"
#include "axle/cg/glmesh.h"
#include "voxel/voxelizer_api.h"

namespace voxel {
//typedef VoxelizeTest<kHost> HostVoxelizeTest;
//typedef VoxelizeTest<kDevice> DeviceVoxelizeTest;

class VoxelizerDemo : public ax::GlutWindow {
private:
  enum { kGL, kBBox, kVoxel };
public:
  VoxelizerDemo(); 
  ~VoxelizerDemo();
  bool Initialize(const char *model_file, int dim, bool uniform);
  void OnIdle();
  void OnPaint();
  void OnResize(int width, int height);
  void OnKeyDown(int key, int x, int y);
private:
  void RunTest();
private:
  ax::OrbitPerspectiveCameraGL camera_;
  ax::FpsCounter fps_counter_;

 /* DeviceVoxelizeTest *d_test_;
  HostVoxelizeTest *h_test_;*/

  voxel::dVoxelizableMeshPtr dmesh_;
  voxel::hVoxelizableMeshPtr hmesh_;
  voxel::Voxels<kDevice> dvoxels_;
  voxel::Voxels<kHost> hvoxels_;
  
  ax::TriMeshPtr mesh_;
  ax::ObjectPtr glmesh_;

  uint32 voxel_list_;
  uint32 bbox_list_;  
  
  int method_;
  int mode_;
  int dim_;
  int iters_;
};
} // namespace voxel

#endif // VOXELIZER_VOXELIZER_TEST_H

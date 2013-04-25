#include "voxel/voxelizer_host.h"
#include "voxelizer_imp.cuh"

#ifdef VOXELIZER_HOST
namespace voxel {
int Voxelize(const float *vertices, const int *triangles, const int N,
             const float *tri_bbox0, const float *tri_bbox1, 
             HostVoxels &vols) {
  // (x, y, z) --> (x + y * n) * (n / 32) + z / 32, and z % 32 index in the byte 
  // check vols data
 /* float d0 = vols.delta()[0];
  float d1 = vols.delta()[1];
  float d2 = vols.delta()[2];
  ax::Logger::Debug("inv_delta_", d0);
  ax::Logger::Debug("inv_delta_", d1);
  ax::Logger::Debug("inv_delta_", d2);*/
  
  for (int i = 0; i < N; ++i) {
    host::voxelize_kernel(i, vertices, triangles, N, tri_bbox0, tri_bbox1,
                          vols.bbox0_ptr(), vols.delta_ptr(),
                          vols.inv_delta_ptr(), vols.stride_ptr(),
                          vols.vols_ptr());
  }

 /* float startxxx = tri_bbox1[2077] * vols.inv_delta_ptr()[0];
  float endxxx = vols.bbox0_ptr()[0] * vols.inv_delta_ptr()[0];
  printf("startxxx: %.12f\n", startxxx);
  printf("endxxx: %.12f\n", endxxx);
  printf("xxxxxx: %.12f\n", startxxx - endxxx);*/

  
  return ax::kOK;
}
} // voxel
#endif // VOXELIZER_HOST

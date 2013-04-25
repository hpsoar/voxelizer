#include "voxel/voxelizer_api.h"
//#include "axle/core/debug.h"

namespace voxel {
static const float epsilon = 0.00001;
bool check_error(const float val1, const float val2) {
  return fabs(val1 - val2) > epsilon;
}

bool CheckVoxels(const DVectorInt &dvols, const HVectorInt &hvols) {
  const HVectorInt dvols2(dvols);
  const int vol_sz = hvols.size();
  for (int i = 0; i < vol_sz; ++i) {
    if (hvols[i] != dvols2[i]) {
      //ax::Logger::Debug("voxelize error");
    }
  }
  return true;
}

__global__ void KernelCopyResult(const int *input, int n, int *output) {
  int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pidx < n) output[pidx] = input[pidx];
}

extern "C" void Voxelize(ax::TriMeshPtr mesh, int dim, tVoxels *voxels) {
  voxel::dVoxelizableMeshPtr dmesh = voxel::ConvertFromTriMesh<kDevice>(mesh);
  ax::SeqTimer::Begin("hello");
  voxel::DeviceVoxels vols;
  dmesh->ComputeTriBBox();
  dmesh->ComputeMeshBBox();
  
  RET(vols.Initialize(HVectorFloat(dmesh->bbox0()), 
                      HVectorFloat(dmesh->bbox1()), 
                      dim, kUniform));
  voxel::Voxelize(thrust::raw_pointer_cast(&dmesh->vertices().front()),
                  thrust::raw_pointer_cast(&dmesh->triangles().front()),
                  dmesh->n_triangles(),
                  thrust::raw_pointer_cast(&dmesh->tri_bbox0().front()),
                  thrust::raw_pointer_cast(&dmesh->tri_bbox1().front()),
                  vols);
  cudaMemcpyKind copyKind = (voxels->target == kDevice ? 
      cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost);

  cudaMemcpy(voxels->data, vols.vols_ptr(), vols.vols().size() * sizeof(int), 
             copyKind);
  cudaMemcpy(voxels->bbox0, vols.bbox0_ptr(), 3 * sizeof(float), copyKind);
  cudaMemcpy(voxels->delta, vols.delta_ptr(), 3 * sizeof(float), copyKind);
  for (int i = 0; i < 3; ++i) voxels->dim[i] = vols.dim(i);
  ax::SeqTimer::End();
}
} // voxel

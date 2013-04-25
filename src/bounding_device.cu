#include "voxel/bounding.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "axle/core/debug.h"

namespace voxel {
__device__ void update_bound(const float f, float *bbox0, float *bbox1) {
  if (f < *bbox0) *bbox0 = f;
  if (f > *bbox1) *bbox1 = f;
}

__device__ void update_bound(const float *vertex, const int tri_idx, const int N,
                             float *bbox0, float *bbox1) {
  int c_idx = tri_idx;
  update_bound(vertex[0], &bbox0[c_idx], &bbox1[c_idx]);
  c_idx += N;
  update_bound(vertex[1], &bbox0[c_idx], &bbox1[c_idx]);
  c_idx += N;
  update_bound(vertex[2], &bbox0[c_idx], &bbox1[c_idx]);
}

__global__ void compute_bounkernel(const float *vertices, const int *triangles,
                                   const int N, float *bbox0, float *bbox1) {
  int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tri_idx >= N) return;

  //bbox0[i] == bbox0[i].x
  //bbox0[i + N] == bbox0[i].y
  //bbox0[i + N + N] == bbox0[i].z
  for (int c = 0, c_idx = tri_idx; c < 3; ++c, c_idx += N) {
    bbox0[c_idx] = INFINITY; 
    bbox1[c_idx] = -INFINITY;
  }
  for (int c = 0, vert_idx = tri_idx; c < 3; ++c, vert_idx += N) {
    update_bound(&vertices[3 * triangles[vert_idx]], tri_idx, N, bbox0, bbox1);
  }
}

int ComputeBoundDevice(const float *vertices, const int *triangles, 
                       const int N, float *tri_bbox0, float *tri_bbox1) {
  assert(NULL != vertices && NULL != triangles &&
         NULL != tri_bbox0 && NULL != tri_bbox1);

  if (NULL == vertices || NULL == triangles || 
      NULL == tri_bbox0 || NULL == tri_bbox1) return ax::kInvalidArg;

  const int block_size = 1024;
  // int block_num = (N + block_size - 1) / block_size;
  int block_num = N / block_size;
  if (N & 0x3ff) ++block_num;
  compute_bounkernel<<<block_num, block_size>>>(vertices, triangles, N, 
                                                  tri_bbox0, tri_bbox1);
  cudaError_t cudaStatus;
  cudaStatus = cudaDeviceSynchronize();
  if (cudaSuccess != cudaStatus ) {
    ax::Logger::Debug("compute bound kernel failed!\n");
    return ax::kFailed;
  }
  return ax::kOK;
}

int UnionBoundDevice(const float *tri_bbox0, const float *tri_bbox1, 
                     const int N, float *bbox0, float *bbox1) {
  //ax::Logger::Log("UnionBoundDevice");
  //CUDPPHandle cudpp_handle;
  //cudppCreate(&cudpp_handle);
  //CUDPPConfiguration config;
  //config.op = CUDPP_MIN;
  //config.datatype = CUDPP_FLOAT;
  //config.algorithm = CUDPP_REDUCE;
  //config.options = CUDPP_OPTION_FORWARD;

  //CUDPPHandle min_reduce_plan = 0, max_reduce_plan = 0;
  //V_RET(CreatePlan(cudpp_handle, config, N, 1, 0, &min_reduce_plan));  

  //config.op = CUDPP_MAX;
  //V_RET(CreatePlan(cudpp_handle, config, N, 1, 0, &max_reduce_plan));  

  //for (int i = 0, offset = 0; i < 3; ++i, offset += N) {
  //  //find mins for every component of tri_bbox0
  //  V_RET(Reduce(min_reduce_plan, tri_bbox0 + offset, N, &bbox0[i]));
  //  
  //  //find maxs for every component of tri_bbox1
  //  V_RET(Reduce(max_reduce_plan, tri_bbox1 + offset, N, &bbox1[i]));
  //}
  return ax::kOK;
}

} // voxel

#include "voxel/voxelizer_device.h"
#include "voxelizer_imp.cuh"

#ifdef VOXELIZER_DEVICE
#include <device_launch_parameters.h>

namespace voxel {
int Voxelize(const float *d_vertices, const int *d_triangles, const int N,
             const float *d_tri_bbox0, const float *d_tri_bbox1, 
             DeviceVoxels &vols) {
  // (x, y, z) --> (x + y * n) * (n / 32) + z / 32, and z % 32 index in the byte   
  const int block_size = 128;
  int block_num = (N + block_size - 1) / block_size;
  thrust::device_vector<int> ttt(N);
  
  device::voxelize_kernel<<<block_num, block_size>>>(
      d_vertices, d_triangles, N, d_tri_bbox0, d_tri_bbox1, vols.bbox0_ptr(),
      vols.delta_ptr(), vols.inv_delta_ptr(), vols.stride_ptr(), vols.vols_ptr()
#ifdef STASTICS_DEVICE
      ,thrust::raw_pointer_cast(&ttt.front())
#endif
      );
 
  cudaError_t cudaStatus;
 /* cudaStatus = cudaDeviceSynchronize();
  if (cudaSuccess != cudaStatus ) {
    ax::Logger::Debug("voxelize kernel failed!\n");
    return kFailed;
  }*/

  /*thrust::device_vector<float> start(N), end(N);
  device::test_kernel<<<block_num, block_size>>>(
      d_tri_bbox0, d_tri_bbox1, vols.inv_delta_ptr(), vols.bbox0_ptr(), N,
      thrust::raw_pointer_cast(&start.front()),
      thrust::raw_pointer_cast(&end.front()));
  cutilCheckMsg("test_kernel()");

  cudaStatus = cudaDeviceSynchronize();
  if (cudaSuccess != cudaStatus ) {
    ax::Logger::Debug("test kernel failed!\n");
    return kFailed;
  }*/

  //float startxxx = start[2077];
  //float endxxx = end[2077];
  //printf("startxxx: %.12f\n", startxxx);
  //printf("endxxx: %.12f\n", endxxx);
  //printf("xxxxxx: %.12f\n", startxxx - endxxx);

  //thrust::device_vector<float> s(128, startxxx);
  //thrust::device_vector<float> e(128, endxxx);
  //thrust::device_vector<float> res(128);
  //device::test2_kernel<<<1, 128>>>(
  //  thrust::raw_pointer_cast(&s.front()), 
  //  thrust::raw_pointer_cast(&e.front()),
  //  thrust::raw_pointer_cast(&res.front()));
  //cutilCheckMsg("test_kernel()");

  //cudaStatus = cudaDeviceSynchronize();
  //if (cudaSuccess != cudaStatus ) {
  //  ax::Logger::Debug("test kernel failed!\n");
  //  return kFailed;
  //}
  //float ss = s[0];
  //float ee = e[0];
  //float r = res[0];
  //printf("startxxx: %.12f\n", ss);
  //printf("endxxx: %.12f\n", ee);
  //printf("xxxxxx: %.12f\n", r);
  //const int n_stastics = 128;
  //int count[n_stastics];
  //memset(count, 0, sizeof(int) * n_stastics);
  //
  //thrust::host_vector<int> h_ttt(ttt);
  //for (int i = 0; i < N; ++i) {
  //  int t = h_ttt[i];
  //  ++count[t];    
  //}
  //ax::Logger::Debug("not preprocessed:", count[0]);
  //ax::Logger::Debug("preprocessed:", count[1]);
  //ax::Logger::Debug("processed:", count[2]);
  //ax::Logger::Debug("simply processed in X:", count[3]);
  //ax::Logger::Debug("simply processed:", count[4]);
  ////ax::Logger::Debug("normal error:", count[10000]);
  return ax::kOK;
}
} // voxel
#endif // VOXELIZER_DEVICE
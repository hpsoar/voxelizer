#if defined(SYS_IS_WINDOWS)
#include <Windows.h>
#endif

#if defined(VOXELIZER_DEVICE)
#include <cuda_runtime.h>
#define _DEVICE_ __device__
#define _CONSTANT_ __constant__
#define _GLOBAL_ __global__
#define NAMESPACE device
#elif defined(VOXELIZER_HOST)
#define _DEVICE_
#define _CONSTANT_
#define _GLOBAL_
#define NAMESPACE host
inline void atomicOr(int *ptr, int val) {
  *ptr |= val;
}
#endif


#include "axle/core.h"

namespace voxel {
namespace NAMESPACE {

_DEVICE_ float dot3(const float *a, const float *b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

_DEVICE_ float *cross(const float *a, const float *b, float *c) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
  return c;
}

_DEVICE_ float dot3(const float *a, const float *b, 
                      const int i1, const int i2) {
  return a[i1] * b[i1] + a[i2] * b[i2];
}

_DEVICE_ float *minus3(const float *a, const float *b, float *c) {
  c[0] = a[0] - b[0];
  c[1] = a[1] - b[1];
  c[2] = a[2] - b[2];
  return c;
}

_DEVICE_ float *minus3(float *a, const float *b) {
  a[0] -= b[0];
  a[1] -= b[1];
  a[2] -= b[2];
  return a;
}

_DEVICE_ void get_tri_vertices(const float *vertices, const int *triangles,
                               const int N,
                               int idx, const float *tri_vertices[3]) {
  tri_vertices[0] = &vertices[3 * triangles[idx]];
  idx += N;
  tri_vertices[1] = &vertices[3 * triangles[idx]];
  idx += N;
  tri_vertices[2] = &vertices[3 * triangles[idx]];
}

_DEVICE_ void get_edges(const float *vertices[3], float edges[3][3]) {
  minus3(vertices[1], vertices[0], edges[0]);
  minus3(vertices[2], vertices[1], edges[1]);
  minus3(vertices[0], vertices[2], edges[2]);
}

// only one component

_DEVICE_ void compute_plane_d(const float *v, const float *delta,
                              const int x, const int y, const int z,
                              float n_d_xy[3]) {
  n_d_xy[z] = -dot3(n_d_xy, v, x, y);

  float tmp = delta[x] * n_d_xy[x];
  if (tmp > 0) n_d_xy[z] += tmp;
  tmp = delta[y] * n_d_xy[y];
  if (tmp > 0) n_d_xy[z] += tmp;
}

_DEVICE_ void compute_plane_n_d_1(const float *v, const float *e, 
                                  const int x, const int y, const int z,
                                  const float *delta, float n_d_xy[3]) {
  n_d_xy[x] = -e[y];
  n_d_xy[y] = e[x];
  compute_plane_d(v, delta, x, y, z, n_d_xy);
}
_DEVICE_ void compute_plane_n_d_0(const float *v, const float *e, 
                                  const int x, const int y, const int z,
                                  const float *delta, float n_d_xy[3]) {
  n_d_xy[x] = e[y];
  n_d_xy[y] = -e[x];
  compute_plane_d(v, delta, x, y, z, n_d_xy);
}

_DEVICE_ void compute_plane_n_d(const float *v[3], const float e[3][3],
                                const int x, const int y, const int z, 
                                const float *n, const float *delta,
                                float n_d_xy[3][3]) {
  if (n[z] < 0) {
    compute_plane_n_d_0(v[0], e[0], x, y, z, delta, n_d_xy[0]);
    compute_plane_n_d_0(v[1], e[1], x, y, z, delta, n_d_xy[1]);
    compute_plane_n_d_0(v[2], e[2], x, y, z, delta, n_d_xy[2]);
  } else {
    compute_plane_n_d_1(v[0], e[0], x, y, z, delta, n_d_xy[0]);
    compute_plane_n_d_1(v[1], e[1], x, y, z, delta, n_d_xy[1]);
    compute_plane_n_d_1(v[2], e[2], x, y, z, delta, n_d_xy[2]);
  }
}

_CONSTANT_ int modulo[] = { 1, 2, 0 };

_DEVICE_ int *VolPtr(int *vols, const int start[3], const int vol_stride[3], int int_begin) {
   return vols + int_begin*vol_stride[Z] + start[Y]*vol_stride[Y] +
                       start[X] * vol_stride[X];
}

_DEVICE_ bool preprocess(const int start[3], const int end[3], 
                         const int int_begin, const int int_end,
                         const int bit_begin, const int bit_end,
                         const int vol_stride[3], int *vols, int *axis) {
  // in an int, 0 index the least sinificant bit
  int flag = 0;
  if (start[X] == end[X]) {
    ++flag;
    *axis = X;
  }

  if (start[Y] == end[Y]) {
    if (flag > 0) {
      // set all along axis Z, and return;
      int *vol_ptr_z = VolPtr(vols, start, vol_stride, int_begin);
      if (int_begin == int_end) {
        // set range [bit_begin, bit_end]
        if (bit_end == 31) {
          int bits = -1 << bit_begin;
          atomicOr(vol_ptr_z, bits);
        } else {
          int bits = (1 << (bit_end+1)) - (1 << bit_begin); 
          atomicOr(vol_ptr_z, bits);
        }
      } else {
        //set range [bit_begin, 31], and range[0, bit_end]
        int bits = -1 << bit_begin; 
        atomicOr(vol_ptr_z, bits);
        for (int i = int_begin + 1; i < int_end; ++i) {
          // set the ints in range (ptr_begin, ptr_end)          
          bits = 0xffffffff;
          vol_ptr_z += vol_stride[Z];
          atomicOr(vol_ptr_z, bits);
        }        
        if (bit_end == 31) {
          bits = 0xffffffff;
        } else {
          bits = (1 << (bit_end + 1)) - 1;
        }
        vol_ptr_z += vol_stride[Z];
        atomicOr(vol_ptr_z, bits);
      }
      return true;
    }
    ++flag;
    *axis = Y;
  }

  if (start[Z] == end[Z]) {
    if (flag > 0) {
      // set all, along axis (1-axis), and return;
      int bits = 1 << bit_begin;

      int *vol = VolPtr(vols, start, vol_stride, int_begin);
      int other_axis = 1 - *axis;
      for (int i = start[other_axis]; i <= end[other_axis]; ++i) {
        // add the bit to *vol;
        atomicOr(vol, bits);
        vol += vol_stride[other_axis];
      }
      return true;
    }
    ++flag;
    *axis = Z;
  }
  return false;
}

_DEVICE_ bool projection_test(const float p[3], const int x, const int y,
                              const int z, const float n_d_ii[3][3]) {
  if ((dot3(n_d_ii[0], p, x, y) + n_d_ii[0][z]) < 0) return false;
  if ((dot3(n_d_ii[1], p, x, y) + n_d_ii[1][z]) < 0) return false;
  if ((dot3(n_d_ii[2], p, x, y) + n_d_ii[2][z]) < 0) return false;
  return true;
}

// [begin, end)
_DEVICE_ int process_range(const int begin, const int end, const float n[3], 
                           float p[3], const float d1, const float d2,
                           const float delta_z, 
                           const float n_d_xy[][3], 
                           const float n_d_yz[][3],
                           const float n_d_zx[][3]) {  
  int bits = 0;
  for (int z = begin; z < end; ++z, p[Z] += delta_z) {
    // evaluate Eq. 1
    float n_dot_p = dot3(n, p);
    if ((n_dot_p + d1) * (n_dot_p + d2) > 0) continue;
   
    if (projection_test(p, Y, Z, X, n_d_yz) &&
        projection_test(p, Z, X, Y, n_d_zx)) {
      // set the bit to val
      int bit = 1 << z;
      bits |= bit;
    }
  }
  return bits;
}

_DEVICE_ int process_range(const int begin, const int end, float p[3],
                             const int x, const int y, const int z,
                             const float delta_z, const float n_d_ii[3][3]) {
  int bits = 0;
  for (int i = begin; i < end; ++i, p[Z] += delta_z) {
    if (projection_test(p, x, y, z, n_d_ii)) {
      // set the bit to val
      bits |= 1 << i;
    }
  }
  return bits;
}
                                

_DEVICE_ void process_simple(const float *v[3], const float e[3][3],
                              const float n[3], const int axis,
                              const float bbox0[3], const float delta[3],
                              const int start[3], const int end[3], 
                              const int int_begin, const int int_end,
                              const int bit_begin, const int bit_end,
                              const int vol_stride[3], int *vols) {
  // only projection test against the plane formed by other two axises
  int axis1 = modulo[axis];
  int axis2 = modulo[axis1];
  float n_d_ii[3][3];
  compute_plane_n_d(v, e, axis1, axis2, axis, n, delta, n_d_ii);

  // iterate over all voxels in the 2d range
  float p[3];
  if (axis == Z) {
    // only one bit is to be set
    int bit = 1 << bit_begin;
    // p[X] = bbox0[X] + start[X] * delta[X];
    p[Y] = bbox0[Y] + start[Y] * delta[Y];
    int *vol_ptr_y = VolPtr(vols, start, vol_stride, int_begin);
    for (int y = start[Y]; y <= end[Y]; ++y) {
      p[X] = bbox0[X] + start[X] * delta[X];
      int *vol_ptr_x = vol_ptr_y;
      for (int x = start[X]; x <= end[X]; ++x) {
        if (projection_test(p, X, Y, Z, n_d_ii)) {          
          // set the bit 
          atomicOr(vol_ptr_x, bit);
        }
        p[X] += delta[X];
        vol_ptr_x += vol_stride[X];
      }
      p[Y] += delta[Y];
      vol_ptr_y += vol_stride[Y];
    }
  } else {
    //axis_i == X or Y
    // process in Z-X or Y-Z plane
    int axis_i = 1 - axis;
    p[axis] = bbox0[axis] + start[axis] * delta[axis];
    p[axis_i] = bbox0[axis_i] + start[axis_i] * delta[axis_i];
    int *vol_ptr = VolPtr(vols, start, vol_stride, int_begin);
    float p_start_z = bbox0[Z] + start[Z] * delta[Z];
    if (int_begin == int_end) {
      for (int i = start[axis_i]; i <= end[axis_i]; ++i) {
        p[Z] = p_start_z;
        int bits = process_range(bit_begin, bit_end + 1, p, axis1, axis2,
                                 axis, delta[Z], n_d_ii);
        atomicOr(vol_ptr, bits);

        p[axis_i] += delta[axis_i];
        vol_ptr += vol_stride[axis_i];
      }
    } else {
      for (int i = start[axis_i]; i <= end[axis_i]; ++i) {
        p[Z] = p_start_z;
        int *vol_ptr_z = vol_ptr;
        int bits = process_range(bit_begin, 32, p, axis1, axis2, axis, 
                                 delta[Z], n_d_ii); 
        atomicOr(vol_ptr_z, bits);
        for (int i = int_begin + 1; i < int_end; ++i) {
          bits = process_range(0, 32, p, axis1, axis2, axis, delta[Z], n_d_ii);
          vol_ptr_z += vol_stride[Z];
          atomicOr(vol_ptr_z, bits);
        }
        bits = process_range(0, bit_end + 1, p, axis1, axis2, axis,
                             delta[Z], n_d_ii);
        vol_ptr_z += vol_stride[Z];
        atomicOr(vol_ptr_z, bits);

        p[axis_i] += delta[axis_i];
        vol_ptr += vol_stride[axis_i];
      }
    }
  }  
}

  // iterate over all voxels in the 3d range
  // for each
      // evaluate Eq. 1
      // continue if > 0
      // evaluate Eq. 3
      // atomic OR if true

_DEVICE_ void compute_d1_d2(const float n[3], const float v0[3], 
                              const float delta[3], float *d1, float *d2) {
  float c[3] = { 0.0, 0.0, 0.0 };
  if (n[X] > 0) c[X] = delta[X];
  if (n[Y] > 0) c[Y] = delta[Y];
  if (n[Z] > 0) c[Z] = delta[Z];
  float tmp[3];
  *d1 = dot3(n, minus3(c, v0, tmp));
  minus3(delta, c, tmp);
  *d2 = dot3(n, minus3(tmp, v0));
}

_DEVICE_ void process(
    const float *v[3], const float e[3][3], const float n[3], 
    const float bbox0[3], const float delta[3], 
    const int start[3], const int end[3],
    const int int_begin, const int int_end, 
    const int bit_begin, const int bit_end, 
    const int *vol_stride, int *vols) {
  // plane test, and all projection tests
  float d1, d2;
  compute_d1_d2(n, v[0], delta, &d1, &d2);

  float n_d_xy[3][3]; //first idx for edge, second (0, 1) for n, (2) for d
  float n_d_yz[3][3]; //first idx for edge, second (1, 2) for n, (0) for d
  float n_d_zx[3][3]; //first idx for edge, second (0, 2) for n, (1) for d

  compute_plane_n_d(v, e, X, Y, Z, n, delta, n_d_xy);
  compute_plane_n_d(v, e, Y, Z, X, n, delta, n_d_yz);
  compute_plane_n_d(v, e, Z, X, Y, n, delta, n_d_zx);

  float p[3], p_start[3];
  p_start[X] = bbox0[X] + start[X] * delta[X];
  p_start[Y] = bbox0[Y] + start[Y] * delta[Y];
  p_start[Z] = bbox0[Z] + start[Z] * delta[Z];
  p[Y] = p_start[Y];
  int *vol_ptr_y = VolPtr(vols, start, vol_stride, int_begin);
  if (int_begin == int_end) {
    for (int y = start[Y]; y <= end[Y]; ++y) {
      p[X] = p_start[X];
      int *vol_ptr_x = vol_ptr_y;

      for (int x = start[X]; x <= end[X]; ++x) {
        if (projection_test(p, X, Y, Z, n_d_xy)) {
          p[Z] = p_start[Z];
          int *vol_ptr_z = vol_ptr_x;
          int bits = process_range(bit_begin, bit_end + 1, n, p, d1, d2, 
                                   delta[Z], n_d_xy, n_d_yz, n_d_zx);
          atomicOr(vol_ptr_z, bits);
        }
        p[X] += delta[X];
        vol_ptr_x += vol_stride[X];
      }
      p[Y] += delta[Y];
      vol_ptr_y += vol_stride[Y];
    }
  } else {
    for (int y = start[Y]; y <= end[Y]; ++y) {
      p[X] = p_start[X];
      int *vol_ptr_x = vol_ptr_y;

      for (int x = start[X]; x <= end[X]; ++x) {
        if (projection_test(p, X, Y, Z, n_d_xy)) {
          p[Z] = p_start[Z];
          int *vol_ptr_z = vol_ptr_x;

          int bits = process_range(bit_begin, 32, n, p, d1, d2, delta[Z],
                                   n_d_xy, n_d_yz, n_d_zx);
          atomicOr(vol_ptr_z, bits);
          for (int i = int_begin + 1; i < int_end; ++i) {
            bits = process_range(0, 32, n, p, d1, d2, delta[Z],
                                 n_d_xy, n_d_yz, n_d_zx);
            vol_ptr_z += vol_stride[Z];
            atomicOr(vol_ptr_z, bits);
          } 
          bits = process_range(0, bit_end + 1, n, p, d1, d2, delta[Z],
                               n_d_xy, n_d_yz, n_d_zx);
          vol_ptr_z += vol_stride[Z];
          atomicOr(vol_ptr_z, bits);
        }
        p[X] += delta[X];
        vol_ptr_x += vol_stride[X];
      }
      p[Y] += delta[Y];
      vol_ptr_y += vol_stride[Y];
    }
  }
}

_DEVICE_ void get_start_end(const float *tri_bbox0, const float *tri_bbox1,
                              const int tri_idx, const int N,
                              const float *inv_delta, const float *bbox0,
                              int start[3], int end[32]) {
  int c_idx = tri_idx;
  start[0] = (tri_bbox0[c_idx] - bbox0[0]) * inv_delta[0];
  end[0] = (tri_bbox1[c_idx] - bbox0[0]) * inv_delta[0];
  c_idx += N;
  start[1] = (tri_bbox0[c_idx] - bbox0[1]) * inv_delta[1];
  end[1] = (tri_bbox1[c_idx] - bbox0[1]) * inv_delta[1];
  c_idx += N;
  start[2] = (tri_bbox0[c_idx] - bbox0[2]) * inv_delta[2];
  end[2] = (tri_bbox1[c_idx] - bbox0[2]) * inv_delta[2];
}

#if defined(VOXELIZER_HOST)
_GLOBAL_ void voxelize_kernel(
    const int tri_idx,
    const float *vertices, const int *triangles, const int N, 
    const float *tri_bbox0, const float *tri_bbox1,
    const float *bbox0, const float *delta, const float *inv_delta,
    const int *vol_stride, int *vols) {
#endif
    
#if defined(VOXELIZER_DEVICE)
#ifdef STASTICS_DEVICE
_GLOBAL_ void voxelize_kernel(   
    const float *vertices, const int *triangles, const int N, 
    const float *tri_bbox0, const float *tri_bbox1,
    const float *bbox0, const float *delta, const float *inv_delta,
    const int *vol_stride, int *vols, int *statsics) {
#else
_GLOBAL_ void voxelize_kernel(   
    const float *vertices, const int *triangles, const int N, 
    const float *tri_bbox0, const float *tri_bbox1,
    const float *bbox0, const float *delta, const float *inv_delta,
    const int *vol_stride, int *vols) {
#endif
  const int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tri_idx >= N) return;
#endif

  int start[3], end[3];
  //range [start, end]
  get_start_end(tri_bbox0, tri_bbox1, tri_idx, N, inv_delta, bbox0, start, end);

  int int_begin = start[Z] >> 5;
  int bit_begin = start[Z] & 31;
  int int_end = end[Z] >> 5;
  int bit_end = end[Z] & 31;

#ifdef STASTICS_DEVICE
  stastics[tri_idx] = 1;
#endif
  int axis = -1;  
  if (preprocess(start, end, int_begin, int_end, bit_begin, bit_end,
                 vol_stride, vols, &axis)) return;
#ifdef STASTICS_DEVICE
  if (axis < 0) { 
    stastics[tri_idx] = 2;
  } else {
    if (X == axis) {
      stastics[tri_idx] = 3;
    } else {
      stastics[tri_idx] = 4;
    }
  }
#endif

  const float *v[3];
  get_tri_vertices(vertices, triangles, N, tri_idx, v);

  float e[3][3];
  get_edges(v, e);

  float n[3];
  cross(e[0], e[1], n); 

  if (axis < 0) {
    process(v, e, n, bbox0, delta, start, end, int_begin, int_end, 
            bit_begin, bit_end, vol_stride, vols);
  } else {
    process_simple(v, e, n, axis, bbox0, delta, start, end, int_begin, 
                   int_end, bit_begin, bit_end, vol_stride, vols);
  }
}

#ifdef VOXELIZER_DEVICE
_GLOBAL_ void test_kernel(const float *tri_bbox0, const float *tri_bbox1,
                          const float *inv_delta, const float *bbox0,
                          const int N, float *start, float *end) {
  const int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tri_idx >= N) return;

  int c_idx = tri_idx;
  start[c_idx] = tri_bbox1[c_idx] * inv_delta[0];
  end[c_idx] = bbox0[0] * inv_delta[0];
}
_GLOBAL_ void test2_kernel(const float *t1, const float *t0, float *o) {
  float tmp1 = t1[threadIdx.x];
  float tmp2 = t0[threadIdx.x];
  o[threadIdx.x] = tmp1 - tmp2;
}
#endif
} // NAMESPACE
} // voxel


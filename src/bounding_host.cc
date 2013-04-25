#include "voxel/bounding.h"

#include <boost/thread.hpp>

#include "axle/core/types.h"
//#include <boost/mpi/collectives.hpp>

namespace voxel {
class MaxOp {
public:
  void operator()(const float f, float *res) const {
    assert(NULL != res);
    if (f > *res) *res = f;
  }
  static const float s_init_val;
};
const float MaxOp::s_init_val = -FLT_MAX;

class MinOp {
public:
  void operator()(const float f, float *res) const {
    assert(NULL != res);
    if (f < *res) *res = f;
  }
  static const float s_init_val;
};
const float MinOp::s_init_val = FLT_MAX;

template<typename Op>
inline int Reduce(const float *h_in, int N, const Op &op, float *h_out) {
  assert(NULL != h_out);
  *h_out = op.s_init_val;
  for (int i = 0; i < N; ++i) {
    op(h_in[i], h_out);
  } 
  return ax::kOK;
}

int UnionBoundHost(const float *h_tri_bbox0, const float *h_tri_bbox1, const int N,
                   float *h_bbox0, float *h_bbox1) {
  if (N < 1) return ax::kOK;
  MinOp min_op;
  MaxOp max_op;
  for (int i = 0, offset = 0; i < 3; ++i, offset += N) {
    Reduce(h_tri_bbox0 + offset, N, min_op, &h_bbox0[i]);
    Reduce(h_tri_bbox1 + offset, N, max_op, &h_bbox1[i]);
  }
  return ax::kOK;
}

void UpdateBound(const float f, const int c_idx, float *bbox0, float *bbox1) {
  if (f < bbox0[c_idx]) bbox0[c_idx] = f;
  if (f > bbox1[c_idx]) bbox1[c_idx] = f;
}

void UpdateBound(const float *vertex, const int tri_idx, const int N, 
                 float *bbox0, float *bbox1) {
  int c_idx = tri_idx;
  UpdateBound(vertex[0], c_idx, bbox0, bbox1);
  c_idx += N;
  UpdateBound(vertex[1], c_idx, bbox0, bbox1);
  c_idx += N;
  UpdateBound(vertex[2], c_idx, bbox0, bbox1);
}

int ComputeBoundSerial(const float *vertices, const int *triangles, 
                       const int N, const int begin, const int end, 
                       float *bbox0, float *bbox1);

class ComputeBoundTask {
public:
  ComputeBoundTask(const float *vertices, const int *triangles, const int N,
                   const int begin, const int end, float *bbox0, float *bbox1)
      : vertices_(vertices), triangles_(triangles), N_(N),
        begin_(begin), end_(end), bbox0_(bbox0), bbox1_(bbox1) { }
  void operator()() {
    ComputeBoundSerial(vertices_, triangles_, N_, begin_, end_, bbox0_, bbox1_);
  }
private:
  const int begin_, end_;
  const float *vertices_;
  const int *triangles_;
  const int N_;
  float *bbox0_;
  float *bbox1_;
};

int ComputeBoundSerial(const float *vertices, const int *triangles, 
                       const int N, const int begin, const int end, 
                       float *bbox0, float *bbox1) {
  for (int tri_idx = begin; tri_idx < end; ++tri_idx) {
    for (int c = 0, c_idx = tri_idx; c < 3; ++c, c_idx += N) {
      bbox0[c_idx] = FLT_MAX; bbox1[c_idx] = -FLT_MAX;
    }
    for (int c = 0, vert_idx = tri_idx; c < 3; ++c, vert_idx += N) {
      UpdateBound(&vertices[3 * triangles[vert_idx]], tri_idx, N, 
                  bbox0, bbox1);
    }
  } 
  return ax::kOK;
}

int ComputeBoundParallel(const float *vertices, const int *triangles, 
                         const int N, const int n_cores, 
                         float *bbox0, float *bbox1) {
  const int n_threads = n_cores;
  const int thread_size = N / n_threads;
  boost::thread_group tg;
  int begin = 0;
  for (int i = 0; i < n_threads - 1; ++i) {
    int end = begin + thread_size;
    tg.create_thread(ComputeBoundTask(vertices, triangles, N, begin, end,
                                      bbox0, bbox1));
    begin = end;
  }
  tg.create_thread(ComputeBoundTask(vertices, triangles, N, begin, N, 
                                    bbox0, bbox1));
  tg.join_all();
  return ax::kOK;
}

int ComputeBoundHost(const float *h_vertices, const int *h_triangles, const int N,
                    float *h_tri_bbox0, float *h_tri_bbox1) {
  int n_cores = 4;
  if (n_cores > 1) {
    return ComputeBoundParallel(h_vertices, h_triangles, N, n_cores,
                                h_tri_bbox0, h_tri_bbox1);
  } else {
    return ComputeBoundSerial(h_vertices, h_triangles, N, 0, N,
                              h_tri_bbox0, h_tri_bbox1);
  }
}

} // voxel

#include "voxelizer_demo.h"

#include <GL/glut.h>

#include "axle/cg/utils.h"
#include "axle/cg/model_gl.h"
#include "axle/core/options.h"
#include "axle/core/utils.h"
#include "axle/core/timer.h"

#include "axle/core/debug.h"
#include "voxel/voxelizer_api.h"

namespace voxel {
VoxelizerDemo::VoxelizerDemo() 
    : ax::GlutWindow("Voxelizer Demo",10, 10, 800, 600, 
                     GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH),
  camera_(glm::vec3(-2.68, 0, 2.55), glm::vec3(0, 0, 0)) {
  camera_.set_min_dist(0.1);
  camera_.set_fovy(30.0);
  camera_.set_velocity(0, 0, 1);
  method_ = kGL;
  
  iters_ = 10;
  mode_ = kUniform;
  dim_ = 256;
}

VoxelizerDemo::~VoxelizerDemo() {
 
}

int DrawVoxel(const HVectorInt &vols, const HVectorFloat &delta,
              const int x_dim, const int y_dim, const int z_dim, const int int_dim) {
  float t[3] = { 0.f, 0.f, 0.f };
  int count = 0;
  const int *ptr = thrust::raw_pointer_cast(&vols.front());
  const int *vol_ptr = ptr;
  
  for (int z = 0; z < int_dim; ++z) {
    t[Y] = 0;
    for (int y = 0; y < y_dim; ++y) {
      t[X] = 0;
      for (int x = 0; x < x_dim; ++x) {
        int bits = *ptr;
        t[Z] = delta[Z] * 32 * z;
        for (int i = 0; i < 32; ++i) {
          if (bits & (1 << i)) {
            glPushMatrix();
            ax::DrawCube(t[0], t[1], t[2], t[0] + delta[0], t[1] + delta[1], t[2] + delta[2]);
            glPopMatrix();
          }
          t[Z] += delta[Z];
        }
        ++ptr;
        t[X] += delta[X];
      }
      t[Y] += delta[Y];
    }
  }
  //for (int y = 0; y < y_dim; ++y) {
  //  t[2] = 0.f;
  //  for (int z = 0; z < z_dim; ++z) {
  //    t[0] = 0.f;
  //    for (int x = 0; x < x_int_dim; ++x) {
  //      int bits = *ptr;
  //      int bit = 1;
  //      for (int i = 0; i < 32; ++i) {
  //        if (bits & bit) {
  //          glPushMatrix();
  //          //[x, y, z * 32 + i]
  //          //glTranslatef(t[0], t[1], t[2]);
  //          ax::DrawCube(t[0], t[1], t[2], t[0] + delta[0], 
  //                       t[1] + delta[1], t[2] + delta[2]);
  //          //glutSolidCube(delta[0]);
  //          glPopMatrix();
  //          ++count;
  //        }
  //        bit <<= 1;
  //        t[0] += delta[0];
  //      }
  //      ++ptr;
  //    }
  //    t[2] += delta[2];
  //  }
  //  t[1] += delta[1];
  //}
  return count;
}

uint32 CreateVoxelDisplayList(const HVectorInt &vols, const HVectorFloat &delta,
                              const HVectorFloat &bbox0, const int x_dim,
                              const int y_dim, const int z_dim, const int int_dim) {
  uint32 voxel_list = glGenLists(1);
  if (voxel_list) {    
    glNewList(voxel_list, GL_COMPILE);    
    glPushMatrix();
    glTranslatef(bbox0[0], -delta[1] * y_dim * 0.5, bbox0[2]);
    int count = DrawVoxel(vols, delta, x_dim, y_dim, z_dim, int_dim);

    /*glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_LINE_BIT);
    glDisable(GL_LIGHTING);
    glLineWidth(1.5);
    glColor3f(0, 0, 0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    DrawVoxel(vols, delta, x_int_dim, y_dim, z_dim);
    glPopAttrib();*/

    glPopMatrix();
    ax::Logger::Debug("voxels:", count);
    glEndList();
  }
  return voxel_list;
}

void DrawBound(const HVectorFloat &tri_bbox0, 
               const HVectorFloat &tri_bbox1, const int N) {
  for (int i = 0; i < N; ++i) {
    ax::DrawCube(tri_bbox0[i], tri_bbox0[i + N], tri_bbox0[i + _2X(N)],
                 tri_bbox1[i], tri_bbox1[i + N], tri_bbox1[i + _2X(N)]);
  }
}
uint32 CreateBoundDisplayList(const HVectorFloat &tri_bbox0, 
                              const HVectorFloat &tri_bbox1, const int N) {
  uint32 bbox_list = glGenLists(1);  
  if (bbox_list) {
    glNewList(bbox_list, GL_COMPILE);    
  /*  glEnable(GL_AUTO_NORMAL);    */
    glColor3f(0, 1, 0);
    DrawBound(tri_bbox0, tri_bbox1, N);

    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_LINE_BIT);
    glDisable(GL_LIGHTING);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth(1.5);
    glColor3f(0, 0, 0);
    DrawBound(tri_bbox0, tri_bbox1, N);
    glPopAttrib();
    /*glDisable(GL_AUTO_NORMAL);  */  
    glEndList();
  }
  return bbox_list;
}

bool VoxelizerDemo::Initialize(const char *model_file, int dim, bool uniform) {
  glewInit();
  ax::Logger::Debug("load model", model_file);

  V_RET(mesh_ = ax::LoadObj(ax::UserOptions::GetFullModelPath(model_file).c_str()));
  V_RET(glmesh_ = ax::GLMesh::Create(mesh_));  
  this->glmesh_->PreProcess(ax::kUseVBO);
  
  V_RET(this->dmesh_ = voxel::ConvertFromTriMesh<kDevice>(this->mesh_));
  V_RET(this->hmesh_ = voxel::ConvertFromTriMesh<kHost>(this->mesh_));  
  
  this->RunTest();
  //bbox0 [p0c0 p1c0 ...][p0c1 p1c1 ...][p0c2 p1c2 ...]
  //bbox1 [p0c0 p1c0 ...][p0c1 p1c1 ...][p0c2 p1c2 ...]
  //pi stands for a point of the bounding box correspond to the ith triangle
  //ci stands for the ith component of the point
  // !!!!!! a test should be taken to show how the initial value is set
 
  return true;
}

void VoxelizerDemo::RunTest() {    
  this->dmesh_->ComputeTriBBox();
  ax::SeqTimer::Begin("bbox");
  this->dmesh_->ComputeMeshBBox();
  ax::SeqTimer::End();
  this->hmesh_->ComputeTriBBox();
  this->hmesh_->ComputeMeshBBox();

  std::vector<float> bbox0(3), bbox1(3);
  cudaMemcpy(&bbox0[0], thrust::raw_pointer_cast(&this->dmesh_->bbox0().front()), sizeof(float)*3, cudaMemcpyDeviceToHost);
  cudaMemcpy(&bbox1[0], thrust::raw_pointer_cast(&this->dmesh_->bbox1().front()), sizeof(float)*3, cudaMemcpyDeviceToHost);

  this->dvoxels_.Initialize(HVectorFloat(this->dmesh_->bbox0()),
                            HVectorFloat(this->dmesh_->bbox1()), 
                            dim_, mode_);
  this->hvoxels_.Initialize(this->hmesh_->bbox0(), this->hmesh_->bbox1(),
                            dim_, mode_);

  //voxel::Voxelize(thrust::raw_pointer_cast(&this->dmesh_->vertices().front()),
  //                thrust::raw_pointer_cast(&this->dmesh_->triangles().front()),
  //                this->dmesh_->n_triangles(),
  //                thrust::raw_pointer_cast(&this->dmesh_->tri_bbox0().front()),
  //                thrust::raw_pointer_cast(&this->dmesh_->tri_bbox1().front()),
  //                this->dvoxels_);
  
  voxel::Voxelize(thrust::raw_pointer_cast(&this->hmesh_->vertices().front()),
                  thrust::raw_pointer_cast(&this->hmesh_->triangles().front()),
                  this->hmesh_->n_triangles(),
                  thrust::raw_pointer_cast(&this->hmesh_->tri_bbox0().front()),
                  thrust::raw_pointer_cast(&this->hmesh_->tri_bbox1().front()),
                  this->hvoxels_);

  /*ax::SeqTimer::Begin("voxel");
  ::tVoxels tvoxels;
  tvoxels.data = this->dvoxels_.vols_ptr();
  tvoxels.target = voxel::kDevice;
  ::Voxelize(this->mesh_, dim_, tvoxels);
  ax::SeqTimer::End();*/

 
  voxel::CheckVoxels(this->dvoxels_.vols(), this->hvoxels_.vols());
  const HostVoxels &vols = this->hvoxels_;
  if (voxel_list_) glDeleteLists(voxel_list_, 1);
  voxel_list_ = CreateVoxelDisplayList(HVectorInt(this->hvoxels_.vols()), 
                                       vols.delta(), vols.bbox0(),
                                       vols.dim(X), vols.dim(Y), vols.dim(Z), vols.int_dim());
 /* if (bbox_list_) glDeleteLists(bbox_list_, 1);
  bbox_list_ = CreateBoundDisplayList(h_test_->tri_bbox0(), 
                                      h_test_->tri_bbox1(),
                                      h_test_->n_triangles());*/
}

void VoxelizerDemo::OnPaint() {  
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glm::mat4 m = camera_.ViewMatrix();
  glLoadMatrixf(&m[0][0]);

 GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
  GLfloat mat_shininess[] = { 50.0 };
  GLfloat mat_diffuse[] = { 0., 0.8, .0, 1.0 };
  glm::vec4 light_position(camera_.position(), 0);  

  glClearColor (0.0, 0.0, 0.0, 0.0);

  // glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
  glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
  // glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
  glLightfv(GL_LIGHT0, GL_POSITION, &light_position[0]);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_DEPTH_TEST);
  glShadeModel(GL_FLAT);

  //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  
  //glutSolidSphere(0.1, 32, 32);
  ax::CheckErrorsGL("hello");
  if (kGL == method_) {  
    glmesh_->Draw(NULL, ax::kNone);
  } else if (kBBox == method_) {
    //glCallList(bbox_list_);
  } else {
    glCallList(voxel_list_);
  }

  glDisable(GL_LIGHT0);
  glDisable(GL_LIGHTING);

  ax::DisplayStatistics("frame rate", fps_counter_.Update(), "fps");
  glutSwapBuffers();
}

void VoxelizerDemo::OnResize(int width, int height) {
  if (width == 0 || height == 0) return;

  glViewport(0, 0, width, height);    

  camera_.set_aspect_ratio(static_cast<double>(width)/height);
  glMatrixMode(GL_PROJECTION);
  glm::mat4 m = camera_.ProjMatrix();
  glLoadMatrixf(&m[0][0]);
}

void VoxelizerDemo::OnIdle() {
  this->RePaint();
}

void VoxelizerDemo::OnKeyDown(const int key, const int x, const int y) {
  switch (key) {
    case 'a':
      camera_.Yaw(-1);
      break;
    case 'd':
      camera_.Yaw(1);
      break;
    case 'w':
      camera_.Walk(1);
      break;
    case 's':
      camera_.Walk(-1);
      break;
    case 'q':
      camera_.Pitch(1);
      break;
    case 'z':
      camera_.Pitch(-1);
      break;
    case 'n':
      method_ = (method_ + 1) % 3;
      break;
    case 'j':
      dim_ <<= 1;
      this->RunTest();
      break;
    case 'k':
      dim_ >>= 1;
      this->RunTest();
      break;
    case 'h':
      if (iters_ > 5) {
        iters_ -= 5;
        this->RunTest();
      }
      break;
    case 'l':
      iters_ += 5;
      this->RunTest();
      break;
    case 'm':
      mode_ = (mode_ + 1) % 3;
      this->RunTest();
      break;
  }
}
} // voxel

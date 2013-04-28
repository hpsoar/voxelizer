set(proj_name VOXELIZER)

cmake_minimum_required(VERSION 2.8)

include(CMakeLists-base.txt)

if (WIN32)
  set(NV_COMPUTE_SDK_ROOT 
     E:/NVIDIA Corporation/NVIDIA GPU Computing SDK 4.0)
else ()
  set(NV_COMPUTE_SDK_ROOT ~/NVIDIA_GPU_Computing_SDK)
endif (WIN32)

include(FindCUDA)
include_directories(${work_dir}/axle/include 
                    ${magic_dir}/glm-0.9.3.B
                    ${NV_COMPUTE_SDK_ROOT}/C/common/inc
                    ${NV_COMPUTE_SDK_ROOT}/shared/inc
                    ./include)

find_package(CUDA REQUIRED)

add_subdirectory(src ${lib_dir})
add_subdirectory(test ${bin_dir})


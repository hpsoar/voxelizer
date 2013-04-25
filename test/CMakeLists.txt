set(VOXELIZER_TEST_SRC main.cu voxelizer_demo.cu voxelizer_test.cu)

LINK_DIRECTORIES(${WORK_DIR}/voxelizer/lib
                 ${WORK_DIR}/axle/lib
                 ${MAGIC_DIR}/cudpp/lib)

#LINK_LIBRARIES("-laxle ")
set(CUDA_NVCC_FLAGS "-arch=sm_20")
set(CMAKE_CXX_FLAGS "-DCHEETAH_DEBUG")
if (WIN32)
  set(LIB_LIST axle glu32 glut32 glew32 devil voxelizer cudpp32)
else ()
  set(LIB_LIST axle GLU glut GLEW IL ILU voxelizer boost_thread cudpp)
endif (WIN32)

CUDA_ADD_EXECUTABLE(voxelizer_test ${VOXELIZER_TEST_SRC})
TARGET_LINK_LIBRARIES(voxelizer_test ${LIB_LIST})
message(STATUS "cuda path = ${CUDA_LIB_PATH}")

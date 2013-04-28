set(VOXELIZER_TEST_SRC main.cu voxelizer_demo.cu voxelizer_test.cu)

link_directories(${work_dir}/voxelizer/lib
                 ${work_dir}/axle/lib)

#LINK_LIBRARIES("-laxle ")
set(CUDA_NVCC_FLAGS "-arch=sm_20")
set(CMAKE_CXX_FLAGS "-DCHEETAH_DEBUG")
if (WIN32)
  set(LIB_LIST glu32 glut32 glew32 devil)
else ()
  set(LIB_LIST GLU glut GLEW IL ILU boost_thread)
endif (WIN32)

set(target_name "test")

CUDA_ADD_EXECUTABLE(${target_name} ${VOXELIZER_TEST_SRC})

set(output "${target_name}/${target_name}")
set(debug_output "${output}_d")

set_target_properties(${target_name} PROPERTIES OUTPUT_NAME_DEBUG ${debug_output})
set_target_properties(${target_name} PROPERTIES OUTPUT_NAME_RELEASE ${output})

set(libs optimized axle optimized voxelizer debug axle_d debug voxelizer_d ${LIB_LIST})
target_link_libraries(${target_name} ${libs})


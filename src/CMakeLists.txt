set(inc_path ../include/voxel)
file(GLOB lib_inc ${inc_path}/*.h)
file(GLOB lib_src *.cc *.cu)

include_directories(${MAGIC_DIR}/cudpp/include)

#set(CUDA_NVCC_FLAGS "-arch=sm_20")

set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)

set(target_name "voxelizer")

cuda_add_library(${target_name} ${LIB_TYPE}
  ${lib_src} ${lib_inc}
  OPTIONS ${GENCODE_SM13} ${GENCODE_SM20}
  )

set(output "${target_name}")
set(debug_output "${output}_d")

set_target_properties(${target_name} PROPERTIES OUTPUT_NAME_DEBUG ${debug_output})
set_target_properties(${target_name} PROPERTIES OUTPUT_NAME_RELEASE ${output})


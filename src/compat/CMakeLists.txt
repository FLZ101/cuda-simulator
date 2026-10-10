add_library(cudart_shim SHARED
  cudart_shim.cpp
)
target_compile_features(cudart_shim PRIVATE cxx_std_20)
set_target_properties(cudart_shim PROPERTIES
  OUTPUT_NAME cudart_shim
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)
cudasim_set_warnings(cudart_shim)
cudasim_apply_sanitizers(cudart_shim)

add_library(cuda_shim SHARED
  cuda_shim.cpp
)
target_compile_features(cuda_shim PRIVATE cxx_std_20)
set_target_properties(cuda_shim PROPERTIES
  OUTPUT_NAME cuda_shim
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)
cudasim_set_warnings(cuda_shim)
cudasim_apply_sanitizers(cuda_shim)

install(TARGETS cudart_shim cuda_shim
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

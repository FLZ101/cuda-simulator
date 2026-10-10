add_executable(vector_add_ref
  smoke/vector_add_ref.cpp
)
target_compile_features(vector_add_ref PRIVATE cxx_std_20)
cudasim_set_warnings(vector_add_ref)
cudasim_apply_sanitizers(vector_add_ref)

add_test(NAME smoke.vector_add_ref COMMAND vector_add_ref)

set(GENERATED_PTX_DIR ${CMAKE_CURRENT_BINARY_DIR}/ptx)
set(GENERATED_VECTOR_ADD_PTX ${GENERATED_PTX_DIR}/vector_add.ptx)

add_custom_command(
  OUTPUT ${GENERATED_VECTOR_ADD_PTX}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_PTX_DIR}
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -ptx -arch=compute_80
          ${CMAKE_CURRENT_SOURCE_DIR}/ptx/vector_add_kernel.cu
          -o ${GENERATED_VECTOR_ADD_PTX}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ptx/vector_add_kernel.cu
  VERBATIM
)
add_custom_target(generate_test_ptx DEPENDS ${GENERATED_VECTOR_ADD_PTX})

add_executable(ptx_parser_smoke
  ptx/parser_smoke.cpp
)
target_compile_features(ptx_parser_smoke PRIVATE cxx_std_20)
target_include_directories(ptx_parser_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ptx_parser_smoke PRIVATE cudasim_core)
target_compile_definitions(ptx_parser_smoke PRIVATE
  PTX_FIXTURE_PATH="${GENERATED_VECTOR_ADD_PTX}"
)
cudasim_set_warnings(ptx_parser_smoke)
cudasim_apply_sanitizers(ptx_parser_smoke)
add_dependencies(ptx_parser_smoke generate_test_ptx)
add_test(NAME ptx.parser_smoke COMMAND ptx_parser_smoke)

add_executable(ptx_fatbin_smoke
  ptx/fatbin_smoke.cpp
)
target_compile_features(ptx_fatbin_smoke PRIVATE cxx_std_20)
target_include_directories(ptx_fatbin_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ptx_fatbin_smoke PRIVATE cudasim_core)
target_compile_definitions(ptx_fatbin_smoke PRIVATE
  PTX_FIXTURE_PATH="${GENERATED_VECTOR_ADD_PTX}"
)
cudasim_set_warnings(ptx_fatbin_smoke)
cudasim_apply_sanitizers(ptx_fatbin_smoke)
add_dependencies(ptx_fatbin_smoke generate_test_ptx)
add_test(NAME ptx.fatbin_smoke COMMAND ptx_fatbin_smoke)

add_executable(vector_add_exec_smoke
  ptx/vector_add_exec_smoke.cpp
)
target_compile_features(vector_add_exec_smoke PRIVATE cxx_std_20)
target_include_directories(vector_add_exec_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(vector_add_exec_smoke PRIVATE cuda_shim)
target_compile_definitions(vector_add_exec_smoke PRIVATE
  PTX_FIXTURE_PATH="${GENERATED_VECTOR_ADD_PTX}"
)
cudasim_set_warnings(vector_add_exec_smoke)
cudasim_apply_sanitizers(vector_add_exec_smoke)
add_dependencies(vector_add_exec_smoke generate_test_ptx)
add_test(NAME ptx.vector_add_exec_smoke COMMAND vector_add_exec_smoke)

add_executable(runtime_api_smoke
  compat/runtime_api_smoke.cpp
)
target_compile_features(runtime_api_smoke PRIVATE cxx_std_20)
target_include_directories(runtime_api_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(runtime_api_smoke PRIVATE cudart_shim)
cudasim_set_warnings(runtime_api_smoke)
cudasim_apply_sanitizers(runtime_api_smoke)
add_test(NAME compat.runtime_api_smoke COMMAND runtime_api_smoke)

add_executable(driver_api_smoke
  compat/driver_api_smoke.cpp
)
target_compile_features(driver_api_smoke PRIVATE cxx_std_20)
target_include_directories(driver_api_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(driver_api_smoke PRIVATE cuda_shim)
cudasim_set_warnings(driver_api_smoke)
cudasim_apply_sanitizers(driver_api_smoke)
add_test(NAME compat.driver_api_smoke COMMAND driver_api_smoke)

add_executable(driver_module_smoke
  compat/driver_module_smoke.cpp
)
target_compile_features(driver_module_smoke PRIVATE cxx_std_20)
target_include_directories(driver_module_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(driver_module_smoke PRIVATE cuda_shim)
target_compile_definitions(driver_module_smoke PRIVATE
  PTX_FIXTURE_PATH="${GENERATED_VECTOR_ADD_PTX}"
)
cudasim_set_warnings(driver_module_smoke)
cudasim_apply_sanitizers(driver_module_smoke)
add_dependencies(driver_module_smoke generate_test_ptx)
add_test(NAME compat.driver_module_smoke COMMAND driver_module_smoke)

add_executable(runtime_registration_smoke
  compat/runtime_registration_smoke.cpp
)
target_compile_features(runtime_registration_smoke PRIVATE cxx_std_20)
target_include_directories(runtime_registration_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(runtime_registration_smoke PRIVATE cudart_shim cudasim_core)
cudasim_set_warnings(runtime_registration_smoke)
cudasim_apply_sanitizers(runtime_registration_smoke)
add_test(NAME compat.runtime_registration_smoke COMMAND runtime_registration_smoke)

add_executable(runtime_stream_event_smoke
  compat/runtime_stream_event_smoke.cpp
)
target_compile_features(runtime_stream_event_smoke PRIVATE cxx_std_20)
target_include_directories(runtime_stream_event_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(runtime_stream_event_smoke PRIVATE cudart_shim)
cudasim_set_warnings(runtime_stream_event_smoke)
cudasim_apply_sanitizers(runtime_stream_event_smoke)
add_test(NAME compat.runtime_stream_event_smoke COMMAND runtime_stream_event_smoke)

add_executable(driver_stream_event_smoke
  compat/driver_stream_event_smoke.cpp
)
target_compile_features(driver_stream_event_smoke PRIVATE cxx_std_20)
target_include_directories(driver_stream_event_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(driver_stream_event_smoke PRIVATE cuda_shim)
cudasim_set_warnings(driver_stream_event_smoke)
cudasim_apply_sanitizers(driver_stream_event_smoke)
add_test(NAME compat.driver_stream_event_smoke COMMAND driver_stream_event_smoke)

add_test(
  NAME compat.exported_symbols
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/compat/check_symbols.sh
          $<TARGET_FILE:cudart_shim>
          $<TARGET_FILE:cuda_shim>
)

add_executable(nvcc_smoke
  compat/nvcc_smoke.cu
)
set_target_properties(nvcc_smoke PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES 80
)
add_test(
  NAME compat.nvcc_smoke
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/compat/run_nvcc_smoke.sh
          $<TARGET_FILE:nvcc_smoke>
          $<TARGET_FILE:cudart_shim>
          $<TARGET_FILE:cuda_shim>
          $<TARGET_FILE_DIR:cudasim_core>
)

add_executable(vector_add_ref
  smoke/vector_add_ref.cpp
)
target_compile_features(vector_add_ref PRIVATE cxx_std_20)
cudasim_set_warnings(vector_add_ref)
cudasim_apply_sanitizers(vector_add_ref)

add_test(NAME smoke.vector_add_ref COMMAND vector_add_ref)

add_executable(memory_model_smoke
  mem/memory_model_smoke.cpp
)
target_compile_features(memory_model_smoke PRIVATE cxx_std_20)
target_include_directories(memory_model_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(memory_model_smoke PRIVATE cudasim_core)
cudasim_set_warnings(memory_model_smoke)
cudasim_apply_sanitizers(memory_model_smoke)
add_test(NAME mem.memory_model_smoke COMMAND memory_model_smoke)

set(GENERATED_PTX_DIR ${CMAKE_CURRENT_BINARY_DIR}/ptx)
set(GENERATED_VECTOR_ADD_PTX ${GENERATED_PTX_DIR}/vector_add.ptx)
set(GENERATED_SHARED_PTX ${GENERATED_PTX_DIR}/shared_memory.ptx)
set(GENERATED_MULTI_WARP_PTX ${GENERATED_PTX_DIR}/multi_warp_barrier.ptx)
set(GENERATED_SAXPY_PTX ${GENERATED_PTX_DIR}/saxpy.ptx)
set(GENERATED_COPY_PTX ${GENERATED_PTX_DIR}/copy.ptx)
set(GENERATED_MATMUL_NAIVE_PTX ${GENERATED_PTX_DIR}/matmul_naive.ptx)
set(GENERATED_MATMUL_TILED_PTX ${GENERATED_PTX_DIR}/matmul_tiled.ptx)
set(GENERATED_REDUCE_SUM_PTX ${GENERATED_PTX_DIR}/reduce_sum.ptx)
set(GENERATED_TRANSPOSE_PTX ${GENERATED_PTX_DIR}/transpose.ptx)
set(GENERATED_RELU_PTX ${GENERATED_PTX_DIR}/relu.ptx)
set(GENERATED_SOFTMAX_PTX ${GENERATED_PTX_DIR}/softmax_row.ptx)

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

add_custom_command(
  OUTPUT ${GENERATED_SHARED_PTX}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_PTX_DIR}
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -ptx -arch=compute_80
          ${CMAKE_CURRENT_SOURCE_DIR}/ptx/shared_memory_kernel.cu
          -o ${GENERATED_SHARED_PTX}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ptx/shared_memory_kernel.cu
  VERBATIM
)
add_custom_target(generate_shared_ptx DEPENDS ${GENERATED_SHARED_PTX})

add_custom_command(
  OUTPUT ${GENERATED_MULTI_WARP_PTX}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_PTX_DIR}
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -ptx -arch=compute_80
          ${CMAKE_CURRENT_SOURCE_DIR}/ptx/multi_warp_barrier_kernel.cu
          -o ${GENERATED_MULTI_WARP_PTX}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ptx/multi_warp_barrier_kernel.cu
  VERBATIM
)
add_custom_target(generate_multi_warp_ptx DEPENDS ${GENERATED_MULTI_WARP_PTX})

add_custom_command(
  OUTPUT ${GENERATED_SAXPY_PTX}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_PTX_DIR}
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -ptx -arch=compute_80
          ${CMAKE_CURRENT_SOURCE_DIR}/kernels/saxpy.cu
          -o ${GENERATED_SAXPY_PTX}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/saxpy.cu
  VERBATIM
)
add_custom_target(generate_saxpy_ptx DEPENDS ${GENERATED_SAXPY_PTX})

add_custom_command(
  OUTPUT ${GENERATED_COPY_PTX}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_PTX_DIR}
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -ptx -arch=compute_80
          ${CMAKE_CURRENT_SOURCE_DIR}/kernels/copy.cu
          -o ${GENERATED_COPY_PTX}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/copy.cu
  VERBATIM
)
add_custom_target(generate_copy_ptx DEPENDS ${GENERATED_COPY_PTX})

foreach(kernel IN ITEMS
    matmul_naive
    matmul_tiled
    reduce_sum
    transpose
    relu
    softmax_row
)
  add_custom_command(
    OUTPUT ${GENERATED_PTX_DIR}/${kernel}.ptx
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_PTX_DIR}
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -ptx -arch=compute_80
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}.cu
            -o ${GENERATED_PTX_DIR}/${kernel}.ptx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}.cu
    VERBATIM
  )
  add_custom_target(generate_${kernel}_ptx ALL
    DEPENDS ${GENERATED_PTX_DIR}/${kernel}.ptx
  )
endforeach()

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

add_executable(shared_memory_exec_smoke
  ptx/shared_memory_exec_smoke.cpp
)
target_compile_features(shared_memory_exec_smoke PRIVATE cxx_std_20)
target_include_directories(shared_memory_exec_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shared_memory_exec_smoke PRIVATE cuda_shim)
target_compile_definitions(shared_memory_exec_smoke PRIVATE
  PTX_FIXTURE_PATH="${GENERATED_SHARED_PTX}"
)
cudasim_set_warnings(shared_memory_exec_smoke)
cudasim_apply_sanitizers(shared_memory_exec_smoke)
add_dependencies(shared_memory_exec_smoke generate_shared_ptx)
add_test(NAME ptx.shared_memory_exec_smoke COMMAND shared_memory_exec_smoke)

add_executable(multi_warp_barrier_exec_smoke
  ptx/multi_warp_barrier_exec_smoke.cpp
)
target_compile_features(multi_warp_barrier_exec_smoke PRIVATE cxx_std_20)
target_include_directories(multi_warp_barrier_exec_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(multi_warp_barrier_exec_smoke PRIVATE cuda_shim)
target_compile_definitions(multi_warp_barrier_exec_smoke PRIVATE
  PTX_FIXTURE_PATH="${GENERATED_MULTI_WARP_PTX}"
)
cudasim_set_warnings(multi_warp_barrier_exec_smoke)
cudasim_apply_sanitizers(multi_warp_barrier_exec_smoke)
add_dependencies(multi_warp_barrier_exec_smoke generate_multi_warp_ptx)
add_test(NAME ptx.multi_warp_barrier_exec_smoke COMMAND multi_warp_barrier_exec_smoke)

add_test(
  NAME ptx.simt_gate
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generated/check_simt.py
          ${GENERATED_VECTOR_ADD_PTX}
          ${GENERATED_SHARED_PTX}
          ${GENERATED_MULTI_WARP_PTX}
          ${GENERATED_SAXPY_PTX}
          ${GENERATED_COPY_PTX}
          ${GENERATED_MATMUL_NAIVE_PTX}
          ${GENERATED_MATMUL_TILED_PTX}
          ${GENERATED_REDUCE_SUM_PTX}
          ${GENERATED_TRANSPOSE_PTX}
          ${GENERATED_RELU_PTX}
          ${GENERATED_SOFTMAX_PTX}
)

add_test(
  NAME ptx.coverage_report
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generated/run_coverage.py
          ${GENERATED_VECTOR_ADD_PTX}
          ${GENERATED_SHARED_PTX}
          ${GENERATED_MULTI_WARP_PTX}
          ${GENERATED_SAXPY_PTX}
          ${GENERATED_COPY_PTX}
          ${GENERATED_MATMUL_NAIVE_PTX}
          ${GENERATED_MATMUL_TILED_PTX}
          ${GENERATED_REDUCE_SUM_PTX}
          ${GENERATED_TRANSPOSE_PTX}
          ${GENERATED_RELU_PTX}
          ${GENERATED_SOFTMAX_PTX}
)

add_executable(diff_vector_add_runner
  differential/fixtures/vector_add/host_runner.cpp
)
target_compile_features(diff_vector_add_runner PRIVATE cxx_std_20)
target_include_directories(diff_vector_add_runner PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(diff_vector_add_runner PRIVATE cuda_shim)
target_compile_definitions(diff_vector_add_runner PRIVATE
  PTX_FIXTURE_PATH="${GENERATED_VECTOR_ADD_PTX}"
)
cudasim_set_warnings(diff_vector_add_runner)
cudasim_apply_sanitizers(diff_vector_add_runner)
add_dependencies(diff_vector_add_runner generate_test_ptx)
add_test(
  NAME differential.vector_add
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/differential/run_all.py
          --runner $<TARGET_FILE:diff_vector_add_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
)

add_executable(demo_runner
  kernels/demo_runner.cpp
)
target_compile_features(demo_runner PRIVATE cxx_std_20)
target_include_directories(demo_runner PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(demo_runner PRIVATE cuda_shim)
target_compile_definitions(demo_runner PRIVATE
  COPY_PTX_PATH="${GENERATED_COPY_PTX}"
  SAXPY_PTX_PATH="${GENERATED_SAXPY_PTX}"
)
cudasim_set_warnings(demo_runner)
cudasim_apply_sanitizers(demo_runner)
add_dependencies(demo_runner generate_copy_ptx generate_saxpy_ptx)
add_test(
  NAME kernels.demo_differential
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/kernels/run_demos.py
          --runner $<TARGET_FILE:demo_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
)

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

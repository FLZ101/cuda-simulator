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

set(CLASSIC_KERNEL_NAMES
  matmul_naive
  matmul_tiled
  reduce_sum
  transpose
  relu
  softmax_row
  conv2d_naive
  conv2d_im2col
  conv2d_shared
  bias_add
  batch_norm
  max_pool2d
  avg_pool2d
  layer_norm
  attention_scores
  embedding_lookup
  gelu
  elementwise_add
  elementwise_mul
  scale
  mean
  dropout
  causal_mask
  residual_add
  rms_norm
  concat
  split
  argmax
  clamp
  neg
  square
  log_softmax
  cross_entropy
  sigmoid
  tanh
  silu
  reciprocal
  exp
  log
  sqrt
  pow
  rsqrt
  minimum
  maximum
  leaky_relu
  abs
  relu6
  softplus
  swiglu
  elu
  exp2
  log2
  floor
  ceil
  rint
  trunc
  acos
  erfc
  hard_sigmoid
  hardswish
  softsign
  log_sigmoid
  subtract
  divide
  copysign
  lerp
  cosh
  sinh
  asin
  atan
  cbrt
  mish
  expm1
  atan2
  hypot
  masked_fill
  alibi_bias
  global_avg_pool
  instance_norm
  pad2d
  upsample_nearest2d
  depthwise_conv2d
  repeat_kv
  channel_shuffle
  pixel_shuffle
  adaptive_avg_pool2d
  local_response_norm
  fused_add_norm
  cumsum
)

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

foreach(kernel IN LISTS CLASSIC_KERNEL_NAMES)
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
  list(APPEND CLASSIC_KERNEL_PTX ${GENERATED_PTX_DIR}/${kernel}.ptx)
endforeach()

set(ALL_GENERATED_PTX
  ${GENERATED_VECTOR_ADD_PTX}
  ${GENERATED_SHARED_PTX}
  ${GENERATED_MULTI_WARP_PTX}
  ${GENERATED_SAXPY_PTX}
  ${GENERATED_COPY_PTX}
  ${CLASSIC_KERNEL_PTX}
)

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
          ${ALL_GENERATED_PTX}
)

add_test(
  NAME ptx.coverage_report
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generated/run_coverage.py
          ${ALL_GENERATED_PTX}
)

add_executable(gpu_reference_runner
  reference/gpu_reference_runner.cu
)
set_target_properties(gpu_reference_runner PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES 80
)
target_include_directories(gpu_reference_runner PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(gpu_reference_runner PRIVATE CUDA::cuda_driver)
target_compile_definitions(gpu_reference_runner PRIVATE
  BIAS_ADD_PTX_PATH="${GENERATED_PTX_DIR}/bias_add.ptx"
  BATCH_NORM_PTX_PATH="${GENERATED_PTX_DIR}/batch_norm.ptx"
  MAX_POOL_PTX_PATH="${GENERATED_PTX_DIR}/max_pool2d.ptx"
  AVG_POOL_PTX_PATH="${GENERATED_PTX_DIR}/avg_pool2d.ptx"
  LAYER_NORM_PTX_PATH="${GENERATED_PTX_DIR}/layer_norm.ptx"
  EMBEDDING_PTX_PATH="${GENERATED_PTX_DIR}/embedding_lookup.ptx"
  TRANSPOSE_PTX_PATH="${GENERATED_PTX_DIR}/transpose.ptx"
  MATMUL_NAIVE_PTX_PATH="${GENERATED_PTX_DIR}/matmul_naive.ptx"
  MATMUL_TILED_PTX_PATH="${GENERATED_PTX_DIR}/matmul_tiled.ptx"
  SOFTMAX_PTX_PATH="${GENERATED_PTX_DIR}/softmax_row.ptx"
  REDUCE_SUM_PTX_PATH="${GENERATED_PTX_DIR}/reduce_sum.ptx"
  CONV2D_NAIVE_PTX_PATH="${GENERATED_PTX_DIR}/conv2d_naive.ptx"
  CONV2D_IM2COL_PTX_PATH="${GENERATED_PTX_DIR}/conv2d_im2col.ptx"
  CONV2D_SHARED_PTX_PATH="${GENERATED_PTX_DIR}/conv2d_shared.ptx"
  ATTENTION_PTX_PATH="${GENERATED_PTX_DIR}/attention_scores.ptx"
  GELU_PTX_PATH="${GENERATED_PTX_DIR}/gelu.ptx"
  COPY_PTX_PATH="${GENERATED_COPY_PTX}"
  SAXPY_PTX_PATH="${GENERATED_SAXPY_PTX}"
  RELU_PTX_PATH="${GENERATED_RELU_PTX}"
  ELEMENTWISE_ADD_PTX_PATH="${GENERATED_PTX_DIR}/elementwise_add.ptx"
  ELEMENTWISE_MUL_PTX_PATH="${GENERATED_PTX_DIR}/elementwise_mul.ptx"
  SCALE_PTX_PATH="${GENERATED_PTX_DIR}/scale.ptx"
  MEAN_PTX_PATH="${GENERATED_PTX_DIR}/mean.ptx"
  DROPOUT_PTX_PATH="${GENERATED_PTX_DIR}/dropout.ptx"
  CAUSAL_MASK_PTX_PATH="${GENERATED_PTX_DIR}/causal_mask.ptx"
  RESIDUAL_ADD_PTX_PATH="${GENERATED_PTX_DIR}/residual_add.ptx"
  RMS_NORM_PTX_PATH="${GENERATED_PTX_DIR}/rms_norm.ptx"
  CONCAT_PTX_PATH="${GENERATED_PTX_DIR}/concat.ptx"
  SPLIT_PTX_PATH="${GENERATED_PTX_DIR}/split.ptx"
  ARGMAX_PTX_PATH="${GENERATED_PTX_DIR}/argmax.ptx"
  CLAMP_PTX_PATH="${GENERATED_PTX_DIR}/clamp.ptx"
  NEG_PTX_PATH="${GENERATED_PTX_DIR}/neg.ptx"
  SQUARE_PTX_PATH="${GENERATED_PTX_DIR}/square.ptx"
  LOG_SOFTMAX_PTX_PATH="${GENERATED_PTX_DIR}/log_softmax.ptx"
  CROSS_ENTROPY_PTX_PATH="${GENERATED_PTX_DIR}/cross_entropy.ptx"
  SIGMOID_PTX_PATH="${GENERATED_PTX_DIR}/sigmoid.ptx"
  TANH_PTX_PATH="${GENERATED_PTX_DIR}/tanh.ptx"
  SILU_PTX_PATH="${GENERATED_PTX_DIR}/silu.ptx"
  RECIPROCAL_PTX_PATH="${GENERATED_PTX_DIR}/reciprocal.ptx"
  EXP_PTX_PATH="${GENERATED_PTX_DIR}/exp.ptx"
  LOG_PTX_PATH="${GENERATED_PTX_DIR}/log.ptx"
  SQRT_PTX_PATH="${GENERATED_PTX_DIR}/sqrt.ptx"
  POW_PTX_PATH="${GENERATED_PTX_DIR}/pow.ptx"
  RSQRT_PTX_PATH="${GENERATED_PTX_DIR}/rsqrt.ptx"
  MINIMUM_PTX_PATH="${GENERATED_PTX_DIR}/minimum.ptx"
  MAXIMUM_PTX_PATH="${GENERATED_PTX_DIR}/maximum.ptx"
  LEAKY_RELU_PTX_PATH="${GENERATED_PTX_DIR}/leaky_relu.ptx"
  ABS_PTX_PATH="${GENERATED_PTX_DIR}/abs.ptx"
  RELU6_PTX_PATH="${GENERATED_PTX_DIR}/relu6.ptx"
  SOFTPLUS_PTX_PATH="${GENERATED_PTX_DIR}/softplus.ptx"
  SWIGLU_PTX_PATH="${GENERATED_PTX_DIR}/swiglu.ptx"
  ELU_PTX_PATH="${GENERATED_PTX_DIR}/elu.ptx"
  EXP2_PTX_PATH="${GENERATED_PTX_DIR}/exp2.ptx"
  LOG2_PTX_PATH="${GENERATED_PTX_DIR}/log2.ptx"
  FLOOR_PTX_PATH="${GENERATED_PTX_DIR}/floor.ptx"
  CEIL_PTX_PATH="${GENERATED_PTX_DIR}/ceil.ptx"
  RINT_PTX_PATH="${GENERATED_PTX_DIR}/rint.ptx"
  TRUNC_PTX_PATH="${GENERATED_PTX_DIR}/trunc.ptx"
  ACOS_PTX_PATH="${GENERATED_PTX_DIR}/acos.ptx"
  ERFC_PTX_PATH="${GENERATED_PTX_DIR}/erfc.ptx"
  HARD_SIGMOID_PTX_PATH="${GENERATED_PTX_DIR}/hard_sigmoid.ptx"
  HARDSWISH_PTX_PATH="${GENERATED_PTX_DIR}/hardswish.ptx"
  SOFTSIGN_PTX_PATH="${GENERATED_PTX_DIR}/softsign.ptx"
  LOG_SIGMOID_PTX_PATH="${GENERATED_PTX_DIR}/log_sigmoid.ptx"
  SUBTRACT_PTX_PATH="${GENERATED_PTX_DIR}/subtract.ptx"
  DIVIDE_PTX_PATH="${GENERATED_PTX_DIR}/divide.ptx"
  COPYSIGN_PTX_PATH="${GENERATED_PTX_DIR}/copysign.ptx"
  LERP_PTX_PATH="${GENERATED_PTX_DIR}/lerp.ptx"
  COSH_PTX_PATH="${GENERATED_PTX_DIR}/cosh.ptx"
  SINH_PTX_PATH="${GENERATED_PTX_DIR}/sinh.ptx"
  ASIN_PTX_PATH="${GENERATED_PTX_DIR}/asin.ptx"
  ATAN_PTX_PATH="${GENERATED_PTX_DIR}/atan.ptx"
  CBRT_PTX_PATH="${GENERATED_PTX_DIR}/cbrt.ptx"
  MISH_PTX_PATH="${GENERATED_PTX_DIR}/mish.ptx"
  EXPM1_PTX_PATH="${GENERATED_PTX_DIR}/expm1.ptx"
  ATAN2_PTX_PATH="${GENERATED_PTX_DIR}/atan2.ptx"
  HYPOT_PTX_PATH="${GENERATED_PTX_DIR}/hypot.ptx"
  MASKED_FILL_PTX_PATH="${GENERATED_PTX_DIR}/masked_fill.ptx"
  ALIBI_BIAS_PTX_PATH="${GENERATED_PTX_DIR}/alibi_bias.ptx"
  GLOBAL_AVG_POOL_PTX_PATH="${GENERATED_PTX_DIR}/global_avg_pool.ptx"
  INSTANCE_NORM_PTX_PATH="${GENERATED_PTX_DIR}/instance_norm.ptx"
  PAD2D_PTX_PATH="${GENERATED_PTX_DIR}/pad2d.ptx"
  UPSAMPLE_NEAREST2D_PTX_PATH="${GENERATED_PTX_DIR}/upsample_nearest2d.ptx"
  DEPTHWISE_CONV2D_PTX_PATH="${GENERATED_PTX_DIR}/depthwise_conv2d.ptx"
  REPEAT_KV_PTX_PATH="${GENERATED_PTX_DIR}/repeat_kv.ptx"
  CHANNEL_SHUFFLE_PTX_PATH="${GENERATED_PTX_DIR}/channel_shuffle.ptx"
  PIXEL_SHUFFLE_PTX_PATH="${GENERATED_PTX_DIR}/pixel_shuffle.ptx"
  ADAPTIVE_AVG_POOL2D_PTX_PATH="${GENERATED_PTX_DIR}/adaptive_avg_pool2d.ptx"
  LOCAL_RESPONSE_NORM_PTX_PATH="${GENERATED_PTX_DIR}/local_response_norm.ptx"
  FUSED_ADD_NORM_PTX_PATH="${GENERATED_PTX_DIR}/fused_add_norm.ptx"
  CUMSUM_PTX_PATH="${GENERATED_PTX_DIR}/cumsum.ptx"
)
add_dependencies(gpu_reference_runner
  generate_bias_add_ptx
  generate_batch_norm_ptx
  generate_max_pool2d_ptx
  generate_avg_pool2d_ptx
  generate_layer_norm_ptx
  generate_embedding_lookup_ptx
  generate_transpose_ptx
  generate_matmul_naive_ptx
  generate_matmul_tiled_ptx
  generate_softmax_row_ptx
  generate_reduce_sum_ptx
  generate_conv2d_naive_ptx
  generate_conv2d_im2col_ptx
  generate_conv2d_shared_ptx
  generate_attention_scores_ptx
  generate_gelu_ptx
  generate_copy_ptx
  generate_saxpy_ptx
  generate_relu_ptx
  generate_elementwise_add_ptx
  generate_elementwise_mul_ptx
  generate_scale_ptx
  generate_mean_ptx
  generate_dropout_ptx
  generate_causal_mask_ptx
  generate_residual_add_ptx
  generate_rms_norm_ptx
  generate_concat_ptx
  generate_split_ptx
  generate_argmax_ptx
  generate_clamp_ptx
  generate_neg_ptx
  generate_square_ptx
  generate_log_softmax_ptx
  generate_cross_entropy_ptx
  generate_sigmoid_ptx
  generate_tanh_ptx
  generate_silu_ptx
  generate_reciprocal_ptx
  generate_exp_ptx
  generate_log_ptx
  generate_sqrt_ptx
  generate_pow_ptx
  generate_rsqrt_ptx
  generate_minimum_ptx
  generate_maximum_ptx
  generate_leaky_relu_ptx
  generate_abs_ptx
  generate_relu6_ptx
  generate_softplus_ptx
  generate_swiglu_ptx
  generate_elu_ptx
  generate_exp2_ptx
  generate_log2_ptx
  generate_floor_ptx
  generate_ceil_ptx
  generate_rint_ptx
  generate_trunc_ptx
  generate_acos_ptx
  generate_erfc_ptx
  generate_hard_sigmoid_ptx
  generate_hardswish_ptx
  generate_softsign_ptx
  generate_log_sigmoid_ptx
  generate_subtract_ptx
  generate_divide_ptx
  generate_copysign_ptx
  generate_lerp_ptx
  generate_cosh_ptx
  generate_sinh_ptx
  generate_asin_ptx
  generate_atan_ptx
  generate_cbrt_ptx
  generate_mish_ptx
  generate_expm1_ptx
  generate_atan2_ptx
  generate_hypot_ptx
  generate_masked_fill_ptx
  generate_alibi_bias_ptx
  generate_global_avg_pool_ptx
  generate_instance_norm_ptx
  generate_pad2d_ptx
  generate_upsample_nearest2d_ptx
  generate_depthwise_conv2d_ptx
  generate_repeat_kv_ptx
  generate_channel_shuffle_ptx
  generate_pixel_shuffle_ptx
  generate_adaptive_avg_pool2d_ptx
  generate_local_response_norm_ptx
  generate_fused_add_norm_ptx
  generate_cumsum_ptx
)
add_test(
  NAME kernels.gpu_reference_differential
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/reference/run_gpu_reference.py
          --runner $<TARGET_FILE:gpu_reference_runner>
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

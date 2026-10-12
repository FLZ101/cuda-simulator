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

set(KERNEL_GROUPS
  elementwise
  convolution
  normalization
  reduction
  attention
  indexing
  loss
  matrix
  misc
  detection
  sequence
)

set(KERNELS_elementwise
  abs
  acos
  asin
  atan
  atan2
  cbrt
  ceil
  clamp
  copy
  copysign
  cosh
  divide
  elementwise_add
  elementwise_mul
  elu
  erfc
  exp
  exp2
  expm1
  floor
  gelu
  hard_sigmoid
  hardswish
  hypot
  leaky_relu
  lerp
  log
  log2
  log_sigmoid
  maximum
  minimum
  mish
  neg
  pow
  reciprocal
  relu
  relu6
  residual_add
  rint
  rsqrt
  saxpy
  scale
  sigmoid
  silu
  sinh
  softplus
  softsign
  sqrt
  square
  subtract
  swiglu
  tanh
  trunc
  hardtanh
  hardshrink
  softshrink
  tanhshrink
  threshold
  prelu
  celu
  selu
  gelu_tanh
  glu
  geglu
  reglu
  sign
  heaviside
  fmod
  remainder
  nextafter
  isfinite
  isinf
  isnan
)

set(KERNELS_convolution
  avg_pool2d
  max_pool2d
  conv2d_naive
  conv2d_im2col
  conv2d_shared
  depthwise_conv2d
  conv_transpose2d
  adaptive_avg_pool2d
  adaptive_max_pool2d
  global_avg_pool
  pad2d
  reflection_pad2d
  upsample_nearest2d
  upsample_bilinear2d
  pixel_shuffle
  pixel_unshuffle
  channel_shuffle
  conv1d
  conv3d
  fold
  unfold
  deform_conv2d
)

set(KERNELS_normalization
  batch_norm
  instance_norm
  layer_norm
  group_norm
  rms_norm
  local_response_norm
  fused_add_norm
  weight_norm
  spectral_norm
)

set(KERNELS_reduction
  reduce_sum
  mean
  argmax
  cumsum
  reduce_prod
  reduce_min
  reduce_max
  argmin
  amax
  amin
  logsumexp
  norm
  std
  var
  cumprod
  cummax
  cummin
  count_nonzero
  all
  any
  median
  quantile
  mode
  searchsorted
)

set(KERNELS_attention
  attention_scores
  softmax_row
  log_softmax
  causal_mask
  dropout
  masked_fill
  alibi_bias
  repeat_kv
  rotary_embedding
  embedding_lookup
  masked_softmax
  scaled_dot_product_attention
  grouped_query_attention
  multi_query_attention
  multi_head_attention_forward
  flash_attention_simt
  sliding_window_attention
  block_sparse_attention
  relative_position_bias
  sinusoidal_position_embedding
  xpos_embedding
  ntk_rope_scaling
  yarn_rope_scaling
  kv_cache_append
  kv_cache_rotate
  kv_cache_update
  paged_kv_cache_read
  paged_kv_cache_write
  fused_mlp
  gated_mlp
  fused_rms_norm_linear
  fused_layer_norm_linear
  cross_attention
  local_attention
)

set(KERNELS_indexing
  concat
  split
  gather
  scatter_add
  where
  select
  one_hot
  index_select
  index_copy
  index_fill
  masked_select
  take
  gather_nd
  scatter
  scatter_reduce
  batch_gather
  argsort
  sort
  topk
  unique
  nonzero
  stack
  tile
  repeat
  repeat_interleave
  expand
  narrow
  slice
  roll
  flip
  rot90
  permute
  reshape
  squeeze
  unsqueeze
  chunk
  unbind
  meshgrid
  swapaxes
)

set(KERNELS_loss
  cross_entropy
  nll_loss
  mse_loss
  l1_loss
  smooth_l1_loss
  huber_loss
  binary_cross_entropy
  binary_cross_entropy_with_logits
  kl_div
  poisson_nll_loss
  hinge_embedding_loss
  cosine_embedding_loss
  margin_ranking_loss
  triplet_margin_loss
  soft_margin_loss
  focal_loss
  cosine_similarity
  ctc_loss
  gaussian_nll_loss
  multilabel_margin_loss
  multimargin_loss
)

set(KERNELS_matrix
  matmul_naive
  matmul_tiled
  transpose
  bias_add
  batch_matmul
  addmm
  outer_product
  dot_product
  einsum
  tensordot
  kron
  mv
  bmm
)

set(KERNELS_misc
  erf
  erfinv
  sinc
  lgamma
  digamma
  softmin
  softmax1d
  spatial_softmax
)

set(KERNELS_detection
  nms
  iou
  ciou
  roi_align
  box_decode
  box_encode
  diou
  giou
  anchor_generate
  batched_nms
  generate_proposals
)

set(KERNELS_sequence
  rnn_cell
  lstm_cell
  gru_cell
  sequence_mask
  beam_search
  ctc_decode
  ctc_prefix_beam_search
  attention_decoder
  transformer_decoder
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

foreach(group IN LISTS KERNEL_GROUPS)
  foreach(kernel IN LISTS KERNELS_${group})
    set(KERNEL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${group}/${kernel}.cu)
    add_custom_command(
      OUTPUT ${GENERATED_PTX_DIR}/${kernel}.ptx
      COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_PTX_DIR}
      COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -ptx -arch=compute_80
              ${KERNEL_SOURCE}
              -o ${GENERATED_PTX_DIR}/${kernel}.ptx
      DEPENDS ${KERNEL_SOURCE}
      VERBATIM
    )
    add_custom_target(generate_${kernel}_ptx ALL
      DEPENDS ${GENERATED_PTX_DIR}/${kernel}.ptx
    )
    list(APPEND CLASSIC_KERNEL_PTX ${GENERATED_PTX_DIR}/${kernel}.ptx)
  endforeach()
endforeach()

set(ALL_GENERATED_PTX
  ${GENERATED_VECTOR_ADD_PTX}
  ${GENERATED_SHARED_PTX}
  ${GENERATED_MULTI_WARP_PTX}
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

add_executable(scalar_math_exec_smoke
  ptx/scalar_math_exec_smoke.cpp
)
target_compile_features(scalar_math_exec_smoke PRIVATE cxx_std_20)
target_include_directories(scalar_math_exec_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(scalar_math_exec_smoke PRIVATE cuda_shim)
target_compile_definitions(scalar_math_exec_smoke PRIVATE
  PTX_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ptx/scalar_math_kernel.ptx"
)
cudasim_set_warnings(scalar_math_exec_smoke)
cudasim_apply_sanitizers(scalar_math_exec_smoke)
add_test(NAME ptx.scalar_math_exec_smoke COMMAND scalar_math_exec_smoke)

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
foreach(group IN LISTS KERNEL_GROUPS)
  foreach(kernel IN LISTS KERNELS_${group})
    string(TOUPPER ${kernel} KERNEL_MACRO)
    target_compile_definitions(gpu_reference_runner PRIVATE
      ${KERNEL_MACRO}_PTX_PATH="${GENERATED_PTX_DIR}/${kernel}.ptx"
    )
  endforeach()
endforeach()

target_compile_definitions(gpu_reference_runner PRIVATE
  GENERATED_PTX_DIR="${GENERATED_PTX_DIR}"
  MAX_POOL_PTX_PATH="${GENERATED_PTX_DIR}/max_pool2d.ptx"
  AVG_POOL_PTX_PATH="${GENERATED_PTX_DIR}/avg_pool2d.ptx"
  EMBEDDING_PTX_PATH="${GENERATED_PTX_DIR}/embedding_lookup.ptx"
  ATTENTION_PTX_PATH="${GENERATED_PTX_DIR}/attention_scores.ptx"
  SOFTMAX_PTX_PATH="${GENERATED_PTX_DIR}/softmax_row.ptx"
)

set(GPU_REFERENCE_PTX_TARGETS "")
foreach(group IN LISTS KERNEL_GROUPS)
  foreach(kernel IN LISTS KERNELS_${group})
    list(APPEND GPU_REFERENCE_PTX_TARGETS generate_${kernel}_ptx)
  endforeach()
endforeach()
add_dependencies(gpu_reference_runner ${GPU_REFERENCE_PTX_TARGETS})
foreach(group IN LISTS KERNEL_GROUPS)
  if(KERNELS_${group})
    add_test(
      NAME kernels.gpu_reference_differential_${group}
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/reference/run_gpu_reference.py
              --group ${group}
              --runner $<TARGET_FILE:gpu_reference_runner>
              --cudart $<TARGET_FILE:cudart_shim>
              --cuda $<TARGET_FILE:cuda_shim>
              --core-dir $<TARGET_FILE_DIR:cudasim_core>
    )
  endif()
endforeach()

set(SIM_PYTHON ${CMAKE_SOURCE_DIR}/.venv/bin/python)

set(GENERATED_MODEL_DIR ${CMAKE_CURRENT_BINARY_DIR}/models/generated)
set(VECTOR_ADD_RUNNER_SRC ${GENERATED_MODEL_DIR}/vector_add_runner.cu)
set(MATMUL_RUNNER_SRC ${GENERATED_MODEL_DIR}/matmul_runner.cu)

add_custom_command(
  OUTPUT ${VECTOR_ADD_RUNNER_SRC}
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
          --spec vector_add
          --output-cpp ${VECTOR_ADD_RUNNER_SRC}
          --n 32
          --block 32
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/_common.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/vector_add.py
  VERBATIM
)

add_executable(vector_add_model_runner ${VECTOR_ADD_RUNNER_SRC})
set_target_properties(vector_add_model_runner PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES 86
)
target_link_libraries(vector_add_model_runner PRIVATE CUDA::cuda_driver)

add_test(
  NAME models.vector_add.runner_cpu
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec vector_add
          --mode cpu
          --runner $<TARGET_FILE:vector_add_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/vector_add
)
add_test(
  NAME models.vector_add.runner_sim
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec vector_add
          --mode gpu
          --simulate
          --compare
          --runner $<TARGET_FILE:vector_add_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/vector_add
)

add_custom_command(
  OUTPUT ${MATMUL_RUNNER_SRC}
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
          --spec matmul
          --output-cpp ${MATMUL_RUNNER_SRC}
          --m 8
          --n 8
          --k 8
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/_common.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/matmul.py
  VERBATIM
)

add_executable(matmul_model_runner ${MATMUL_RUNNER_SRC})
set_target_properties(matmul_model_runner PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES 86
)
target_link_libraries(matmul_model_runner PRIVATE CUDA::cuda_driver)

add_test(
  NAME models.matmul.runner_cpu
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec matmul
          --mode cpu
          --runner $<TARGET_FILE:matmul_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/matmul
)
add_test(
  NAME models.matmul.runner_sim
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec matmul
          --mode gpu
          --simulate
          --compare
          --runner $<TARGET_FILE:matmul_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/matmul
)

set(MLP_RUNNER_SRC ${GENERATED_MODEL_DIR}/mlp_runner.cu)

add_custom_command(
  OUTPUT ${MLP_RUNNER_SRC}
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
          --spec mlp
          --output-cpp ${MLP_RUNNER_SRC}
          --batch 4
          --in-features 16
          --hidden 32
          --out-features 8
          --block 32
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/_common.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/mlp.py
  VERBATIM
)

add_executable(mlp_model_runner ${MLP_RUNNER_SRC})
set_target_properties(mlp_model_runner PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES 86
)
target_link_libraries(mlp_model_runner PRIVATE CUDA::cuda_driver)

add_test(
  NAME models.mlp.runner_cpu
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec mlp
          --mode cpu
          --runner $<TARGET_FILE:mlp_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/mlp
)
add_test(
  NAME models.mlp.runner_sim
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec mlp
          --mode gpu
          --simulate
          --compare
          --runner $<TARGET_FILE:mlp_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/mlp
)

set(MODULE_RUNNER_SRC ${GENERATED_MODEL_DIR}/module_runner.cu)
set(MODULE_MODEL_FIXTURE
  ${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures/mlp_module.py)

add_custom_command(
  OUTPUT ${MODULE_RUNNER_SRC}
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
          --spec module
          --output-cpp ${MODULE_RUNNER_SRC}
          --module-file ${MODULE_MODEL_FIXTURE}
          --batch 4
          --in-features 16
          --hidden 32
          --out-features 8
          --block 32
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/_common.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/module.py
    ${MODULE_MODEL_FIXTURE}
  VERBATIM
)

add_executable(module_model_runner ${MODULE_RUNNER_SRC})
set_target_properties(module_model_runner PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES 86
)
target_link_libraries(module_model_runner PRIVATE CUDA::cuda_driver)

add_test(
  NAME models.module.runner_cpu
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec module
          --mode cpu
          --module-file ${MODULE_MODEL_FIXTURE}
          --runner $<TARGET_FILE:module_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/module
)
add_test(
  NAME models.module.runner_sim
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec module
          --mode gpu
          --simulate
          --compare
          --module-file ${MODULE_MODEL_FIXTURE}
          --runner $<TARGET_FILE:module_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/module
)

set(CONV_MODULE_RUNNER_SRC ${GENERATED_MODEL_DIR}/conv_module_runner.cu)
set(CONV_MODEL_FIXTURE
  ${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures/conv_module.py)

add_custom_command(
  OUTPUT ${CONV_MODULE_RUNNER_SRC}
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
          --spec module
          --output-cpp ${CONV_MODULE_RUNNER_SRC}
          --module-file ${CONV_MODEL_FIXTURE}
          --batch 1
          --in-channels 3
          --out-channels 4
          --height 8
          --width 8
          --block 32
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/_common.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/module.py
    ${CONV_MODEL_FIXTURE}
  VERBATIM
)

add_executable(conv_module_model_runner ${CONV_MODULE_RUNNER_SRC})
set_target_properties(conv_module_model_runner PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES 86
)
target_link_libraries(conv_module_model_runner PRIVATE CUDA::cuda_driver)

add_test(
  NAME models.conv_module.runner_cpu
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec module
          --mode cpu
          --module-file ${CONV_MODEL_FIXTURE}
          --batch 1
          --in-channels 3
          --out-channels 4
          --height 8
          --width 8
          --runner $<TARGET_FILE:conv_module_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/conv_module
)
add_test(
  NAME models.conv_module.runner_sim
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec module
          --mode gpu
          --simulate
          --compare
          --module-file ${CONV_MODEL_FIXTURE}
          --batch 1
          --in-channels 3
          --out-channels 4
          --height 8
          --width 8
          --runner $<TARGET_FILE:conv_module_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/conv_module
)

set(RESNET_STEM_RUNNER_SRC ${GENERATED_MODEL_DIR}/resnet_stem_runner.cu)
set(RESNET_STEM_MODEL_FIXTURE
  ${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures/resnet_stem.py)

add_custom_command(
  OUTPUT ${RESNET_STEM_RUNNER_SRC}
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
          --spec module
          --output-cpp ${RESNET_STEM_RUNNER_SRC}
          --module-file ${RESNET_STEM_MODEL_FIXTURE}
          --batch 1
          --in-channels 3
          --out-channels 4
          --height 8
          --width 8
          --block 32
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/_common.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/module.py
    ${RESNET_STEM_MODEL_FIXTURE}
  VERBATIM
)

add_executable(resnet_stem_model_runner ${RESNET_STEM_RUNNER_SRC})
set_target_properties(resnet_stem_model_runner PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES 86
)
target_link_libraries(resnet_stem_model_runner PRIVATE CUDA::cuda_driver)

add_test(
  NAME models.resnet_stem.runner_cpu
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec module
          --mode cpu
          --module-file ${RESNET_STEM_MODEL_FIXTURE}
          --batch 1
          --in-channels 3
          --out-channels 4
          --height 8
          --width 8
          --runner $<TARGET_FILE:resnet_stem_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/resnet_stem
)
add_test(
  NAME models.resnet_stem.runner_sim
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec module
          --mode gpu
          --simulate
          --compare
          --module-file ${RESNET_STEM_MODEL_FIXTURE}
          --batch 1
          --in-channels 3
          --out-channels 4
          --height 8
          --width 8
          --runner $<TARGET_FILE:resnet_stem_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/resnet_stem
)

set(CNN_OPS_RUNNER_SRC ${GENERATED_MODEL_DIR}/cnn_ops_runner.cu)
set(CNN_OPS_MODEL_FIXTURE
  ${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures/cnn_ops.py)

add_custom_command(
  OUTPUT ${CNN_OPS_RUNNER_SRC}
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
          --spec module
          --output-cpp ${CNN_OPS_RUNNER_SRC}
          --module-file ${CNN_OPS_MODEL_FIXTURE}
          --batch 1
          --in-channels 3
          --out-channels 4
          --height 8
          --width 8
          --block 32
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/_common.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/module.py
    ${CNN_OPS_MODEL_FIXTURE}
  VERBATIM
)

add_executable(cnn_ops_model_runner ${CNN_OPS_RUNNER_SRC})
set_target_properties(cnn_ops_model_runner PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES 86
)
target_link_libraries(cnn_ops_model_runner PRIVATE CUDA::cuda_driver)

add_test(
  NAME models.cnn_ops.runner_cpu
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec module
          --mode cpu
          --module-file ${CNN_OPS_MODEL_FIXTURE}
          --batch 1
          --in-channels 3
          --out-channels 4
          --height 8
          --width 8
          --runner $<TARGET_FILE:cnn_ops_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/cnn_ops
)
add_test(
  NAME models.cnn_ops.runner_sim
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec module
          --mode gpu
          --simulate
          --compare
          --module-file ${CNN_OPS_MODEL_FIXTURE}
          --batch 1
          --in-channels 3
          --out-channels 4
          --height 8
          --width 8
          --runner $<TARGET_FILE:cnn_ops_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/cnn_ops
)

set(LLM_OPS_RUNNER_SRC ${GENERATED_MODEL_DIR}/llm_ops_runner.cu)
set(LLM_OPS_MODEL_FIXTURE
  ${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures/llm_ops.py)

add_custom_command(
  OUTPUT ${LLM_OPS_RUNNER_SRC}
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
          --spec module
          --output-cpp ${LLM_OPS_RUNNER_SRC}
          --module-file ${LLM_OPS_MODEL_FIXTURE}
          --batch 4
          --in-features 8
          --block 32
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/models/generate_model_runner.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/_common.py
    ${CMAKE_CURRENT_SOURCE_DIR}/models/specs/module.py
    ${LLM_OPS_MODEL_FIXTURE}
  VERBATIM
)

add_executable(llm_ops_model_runner ${LLM_OPS_RUNNER_SRC})
set_target_properties(llm_ops_model_runner PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES 86
)
target_link_libraries(llm_ops_model_runner PRIVATE CUDA::cuda_driver)

add_test(
  NAME models.llm_ops.runner_cpu
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec module
          --mode cpu
          --module-file ${LLM_OPS_MODEL_FIXTURE}
          --batch 4
          --in-features 8
          --runner $<TARGET_FILE:llm_ops_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/llm_ops
)
add_test(
  NAME models.llm_ops.runner_sim
  COMMAND ${SIM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/models/model_runner.py
          --spec module
          --mode gpu
          --simulate
          --compare
          --module-file ${LLM_OPS_MODEL_FIXTURE}
          --batch 4
          --in-features 8
          --runner $<TARGET_FILE:llm_ops_model_runner>
          --cudart $<TARGET_FILE:cudart_shim>
          --cuda $<TARGET_FILE:cuda_shim>
          --core-dir $<TARGET_FILE_DIR:cudasim_core>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/models/llm_ops
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

add_executable(vector_add_ref
  smoke/vector_add_ref.cpp
)
target_compile_features(vector_add_ref PRIVATE cxx_std_20)
cudasim_set_warnings(vector_add_ref)
cudasim_apply_sanitizers(vector_add_ref)

add_test(NAME smoke.vector_add_ref COMMAND vector_add_ref)

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

add_test(
  NAME compat.exported_symbols
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/compat/check_symbols.sh
          $<TARGET_FILE:cudart_shim>
          $<TARGET_FILE:cuda_shim>
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/version_config.h
  @ONLY
)

add_library(cudasim_core SHARED
  compat_state.cpp
  version.cpp
  ../runtime/context.cpp
  ../runtime/device.cpp
  ../runtime/event.cpp
  ../runtime/memory.cpp
  ../runtime/runtime.cpp
  ../runtime/stream.cpp
  ../ptx/token.cpp
  ../ptx/lexer.cpp
  ../ptx/parser.cpp
  ../ptx/ir.cpp
  ../ptx/lower.cpp
  ../ptx/module.cpp
  ../loader/fatbin.cpp
  ../loader/module_registry.cpp
  ../exec/executor.cpp
  ../jit/ptx_to_cpp.cpp
  ../jit/native_backend.cpp
  ../mem/device_memory.cpp
  ../mem/coalescer.cpp
  ../mem/atomics.cpp
)

target_include_directories(cudasim_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_features(cudasim_core PUBLIC cxx_std_20)
set_target_properties(cudasim_core PROPERTIES
  OUTPUT_NAME cudasim_core
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)
cudasim_set_warnings(cudasim_core)
cudasim_apply_sanitizers(cudasim_core)

install(TARGETS cudasim_core
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

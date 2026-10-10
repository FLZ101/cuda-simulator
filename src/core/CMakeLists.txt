configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/version_config.h
  @ONLY
)

add_library(cudasim_core SHARED
  compat_state.cpp
  version.cpp
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

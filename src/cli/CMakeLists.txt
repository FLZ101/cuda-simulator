add_executable(cuda_sim_run
  main.cpp
)
target_compile_features(cuda_sim_run PRIVATE cxx_std_20)
target_link_libraries(cuda_sim_run PRIVATE cudasim_core)
cudasim_set_warnings(cuda_sim_run)
cudasim_apply_sanitizers(cuda_sim_run)

install(TARGETS cuda_sim_run
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

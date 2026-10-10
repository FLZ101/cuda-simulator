cmake_minimum_required(VERSION 3.24)

project(cuda-simulator VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(cmake/CompilerWarnings.cmake)
include(cmake/SimulatorOptions.cmake)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  message(STATUS "Found CUDA toolkit ${CUDAToolkit_VERSION}")
  message(STATUS "nvcc: ${CUDAToolkit_NVCC_EXECUTABLE}")
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
  enable_language(CUDA)
else()
  message(STATUS "CUDA toolkit not found; nvcc smoke test disabled")
endif()

option(CUDA_SIM_ENABLE_JIT "Enable the optional LLVM JIT backend" OFF)
option(CUDA_SIM_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(CUDA_SIM_ENABLE_JIT)
  find_package(LLVM REQUIRED CONFIG)
  message(STATUS "CUDA simulator JIT enabled with LLVM ${LLVM_PACKAGE_VERSION}")
endif()

add_subdirectory(src)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

cmake_minimum_required(VERSION 3.24)

project(cuda-simulator VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(cmake/CompilerWarnings.cmake)
include(cmake/SimulatorOptions.cmake)

find_package(Python3 3.10 REQUIRED COMPONENTS Development)
message(STATUS "Found Python development headers ${Python3_VERSION}")

find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA toolkit ${CUDAToolkit_VERSION}")
message(STATUS "nvcc: ${CUDAToolkit_NVCC_EXECUTABLE}")

find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi REQUIRED)
execute_process(
  COMMAND ${NVIDIA_SMI_EXECUTABLE} -L
  RESULT_VARIABLE NVIDIA_SMI_RESULT
  OUTPUT_VARIABLE NVIDIA_SMI_OUTPUT
  ERROR_QUIET
)
if(NOT NVIDIA_SMI_RESULT EQUAL 0 OR NVIDIA_SMI_OUTPUT STREQUAL "")
  message(FATAL_ERROR "A CUDA-capable GPU is required for this project")
endif()
string(REGEX REPLACE "\n.*" "" NVIDIA_GPU_NAME "${NVIDIA_SMI_OUTPUT}")
message(STATUS "GPU detected: ${NVIDIA_GPU_NAME}")

set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
enable_language(CUDA)

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

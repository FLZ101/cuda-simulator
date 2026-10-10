find_package(LLVM CONFIG QUIET)

if(NOT LLVM_FOUND)
  set(LLVM_FOUND FALSE)
endif()

if(LLVM_FOUND)
  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
else()
  message(STATUS "LLVM not found; JIT backend will be disabled")
endif()

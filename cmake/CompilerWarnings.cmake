function(cudasim_set_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wsign-conversion
      -Wshadow
      -Werror
    )
  elseif(MSVC)
    target_compile_options(${target} PRIVATE
      /W4
      /WX
    )
  endif()
endfunction()

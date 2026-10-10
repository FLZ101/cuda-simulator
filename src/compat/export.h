#pragma once

#if defined(_WIN32)
#define CUDASIM_API __declspec(dllexport)
#else
#define CUDASIM_API __attribute__((visibility("default")))
#endif

#define CUDASIM_EXTERN_C extern "C"

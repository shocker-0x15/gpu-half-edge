#pragma once

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#    define HP_Platform_Windows
#    if defined(_MSC_VER)
#        define HP_Platform_Windows_MSVC
#    endif
#elif defined(__APPLE__)
#    define HP_Platform_macOS
#endif

#ifdef _DEBUG
#   define ENABLE_ASSERT 1
#   define DEBUG_SELECT(A, B) A
#else
#   define ENABLE_ASSERT 0
#   define DEBUG_SELECT(A, B) B
#endif

#if defined(HP_Platform_Windows_MSVC)
#   define WIN32_LEAN_AND_MEAN
#   define NOMINMAX
#   define _USE_MATH_DEFINES
#   include <Windows.h>
#   undef WIN32_LEAN_AND_MEAN
#   undef NOMINMAX
#   undef near
#   undef far
#   undef RGB
#endif

#if defined(__CUDA_ARCH__)
#else
#   include <cstdio>
#   include <cstdlib>
#   include <cstdint>
#   include <cstdarg>
#   include <cassert>
#   include <string>
#   include <string_view>
#   include <vector>
#   include <filesystem>

#   include <immintrin.h>
#endif

#include "cuda_util.h"



#if defined(__CUDA_ARCH__)
#   define devPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#   if defined(HP_Platform_Windows_MSVC)
void devPrintf(const char* fmt, ...);
#       define hpprintf(fmt, ...) do { devPrintf(fmt, ##__VA_ARGS__); printf(fmt, ##__VA_ARGS__); } while (0)
#   else
#       define devPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#       define hpprintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#   endif
#endif

#if defined(__CUDA_ARCH__)
#   define __Assert(expr, fmt, ...) do { if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); assert(false); } } while (0)
#else
#   define __Assert(expr, fmt, ...) do { if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); abort(); } } while (0)
#endif

#ifdef ENABLE_ASSERT
#   define Assert(expr, fmt, ...) __Assert(expr, fmt, ##__VA_ARGS__)
#else
#   define Assert(expr, fmt, ...)
#endif

#define Assert_Release(expr, fmt, ...) __Assert(expr, fmt, ##__VA_ARGS__)

#define Assert_ShouldNotBeCalled() __Assert(false, "Should not be called!")
#define Assert_NotImplemented() __Assert(false, "Not implemented yet!")



template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE T alignUp(T value, uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t tzcnt(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __clz(__brev(x));
#else
    return _tzcnt_u32(x);
#endif
}

CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t lzcnt(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __clz(x);
#else
    return _lzcnt_u32(x);
#endif
}

CUDA_COMMON_FUNCTION CUDA_INLINE int32_t popcnt(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __popc(x);
#else
    return _mm_popcnt_u32(x);
#endif
}

//     0: 0
//     1: 0
//  2- 3: 1
//  4- 7: 2
//  8-15: 3
// 16-31: 4
// ...
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t prevPowOf2Exp(uint32_t x) {
    if (x == 0)
        return 0;
    return 31 - lzcnt(x);
}

//    0: 0
//    1: 0
//    2: 1
// 3- 4: 2
// 5- 8: 3
// 9-16: 4
// ...
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t nextPowOf2Exp(uint32_t x) {
    if (x == 0)
        return 0;
    return 32 - lzcnt(x - 1);
}

//     0: 0
//     1: 1
//  2- 3: 2
//  4- 7: 4
//  8-15: 8
// 16-31: 16
// ...
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t prevPowOf2(uint32_t x) {
    if (x == 0)
        return 0;
    return 1 << prevPowOf2Exp(x);
}

//    0: 0
//    1: 1
//    2: 2
// 3- 4: 4
// 5- 8: 8
// 9-16: 16
// ...
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t nextPowOf2(uint32_t x) {
    if (x == 0)
        return 0;
    return 1 << nextPowOf2Exp(x);
}



#if !defined(__CUDA_ARCH__)

std::filesystem::path getExecutableDirectory();

#endif // #if !defined(__CUDA_ARCH__)

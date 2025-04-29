set(TARGET_NAME "gpu-half-edge")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp
)

file(
    GLOB CUDA_KERNELS
    "gpu_kernels.cu"
)

file(
    GLOB CUDA_SOURCES
    "misc.cu"
)

nvcuda_compile_ptx(
    SOURCES ${CUDA_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES CUDA_PTXES
    NVCC_OPTIONS
    "-Xcompiler" "\"/wd 4819 /Zc:__cplusplus\""
    "$<$<CONFIG:Debug>:-D_DEBUG=1>"
    "$<$<CONFIG:Debug>:-G>"
    "--gpu-architecture=compute_75"
    "-std=c++20"
    "-cudart" "shared"
    "--use_fast_math"
)

set_common_sources()

# essentials
source_group(
    "essentials" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/[^/]*\.(h|hpp|c|cpp|cu)$")

list(APPEND CUDA_NVCC_FLAGS "$<$<CONFIG:Debug>:-G>")

add_executable(
    "${TARGET_NAME}"
    ${COMMON_SOURCES}
    ${SOURCES}
    ${CUDA_KERNELS}
    ${CUDA_SOURCES}
)
target_compile_definitions(
    "${TARGET_NAME}" PRIVATE
    "USE_CUBD_LIB"
    "CUDA_UTIL_DONT_USE_GL_INTEROP"
)
target_compile_options(
    "${TARGET_NAME}" PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>"
)
target_include_directories(
    "${TARGET_NAME}" PRIVATE
    "../../ext/assimp/include"
    "${CMAKE_BINARY_DIR}/ext/assimp/include"
)
target_link_libraries(
    "${TARGET_NAME}"
    fakelib
    CUDA::cuda_driver
    libcubd_static
    assimp::assimp
)

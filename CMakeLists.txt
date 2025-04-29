cmake_minimum_required(VERSION 3.18)
project(
    gpu-half-edge_cmake
    VERSION 1.0
    DESCRIPTION "Fully parallel construction of a half-edge data structure on GPU"
    LANGUAGES C CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_LIBRARY_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(
    CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake"
    ${CMAKE_MODULE_PATH})
find_package(CUDAToolkit 12.0 REQUIRED)
include("copy_files")
include("nvcuda_compile_ptx")



# ----------------------------------------------------------------
# JP: CMakeLists.txtのあるサブディレクトリ内のターゲット全てにフォルダーを設定する。
# https://stackoverflow.com/questions/45092198/cmake-how-do-i-change-properties-on-subdirectory-project-targets

# _dir以下の(CMakeが有効な)サブディレクトリにあるターゲットを再帰的に取得する。
function(get_all_targets _result _dir)
    get_property(_subdirs DIRECTORY "${_dir}" PROPERTY SUBDIRECTORIES)
    foreach(_subdir IN LISTS _subdirs)
        get_all_targets(${_result} "${_subdir}")
    endforeach()
    get_property(_sub_targets DIRECTORY "${_dir}" PROPERTY BUILDSYSTEM_TARGETS)
    set(${_result} ${${_result}} ${_sub_targets} PARENT_SCOPE)
endfunction()

# _dir以下の(CMakeが有効な)サブディレクトリにあるターゲットにフォルダーを設定する。
function(set_project_group _folder_name _dir)
    get_all_targets(_targets "${_dir}")
    foreach(_target IN LISTS _targets)
        # message(${_target})
        set_target_properties(
            ${_target}
            PROPERTIES FOLDER "${_folder_name}"
        )
    endforeach()
endfunction()

# END: JP: CMakeLists.txtのあるサブディレクトリ内のターゲット全てにフォルダーを設定する。
# ----------------------------------------------------------------



# CUBd
option(CUBD_BUILD_SAMPLES "Build sample projects for cubd." OFF)
option(CUBD_BUILD_DYNAMIC_LIB "Build dynamic-link cubd library." OFF)
option(CUBD_BUILD_STATIC_LIB "Build static-link cubd library." ON)
add_subdirectory(ext/cubd)
set_project_group("External" "ext/cubd")

# Assimp
option(ASSIMP_BUILD_ZLIB "Build your own zlib" ON)
option(ASSIMP_BUILD_ASSIMP_TOOLS "If the supplementary tools for Assimp are built in addition to the library." OFF)
option(ASSIMP_BUILD_SAMPLES "If the official samples are built as well (needs Glut)." OFF)
option(ASSIMP_BUILD_TESTS "If the test suite for Assimp is built in addition to the library." OFF)
option(ASSIMP_INSTALL "Disable this if you want to use assimp as a submodule." OFF)
if(MSVC)
    option(ASSIMP_BUILD_ASSIMP_VIEW "If the Assimp view tool is built. (requires DirectX)" OFF)
endif()
add_subdirectory(ext/assimp)
set_project_group("External" "ext/assimp")



file(
    GLOB COMMON_SOURCES
    "${CMAKE_SOURCE_DIR}/common/cuda_util.h"
    "${CMAKE_SOURCE_DIR}/common/cuda_util.cpp"
    "${CMAKE_SOURCE_DIR}/common/common.h"
    "${CMAKE_SOURCE_DIR}/common/common.cpp"
)

# Called from the sub directories
function(set_common_sources)
    # non-essentials
    source_group(
        "non-essentials" REGULAR_EXPRESSION
        "${CMAKE_SOURCE_DIR}/common/[^/]*\.(h|hpp|c|cpp)$")
endfunction()



add_library(fakelib INTERFACE)
target_compile_definitions(
    fakelib INTERFACE
    "$<$<CONFIG:Debug>:_DEBUG=1>"
)
target_compile_options(
    fakelib INTERFACE
    # if (compilerID == MSVC && compilerLanguage != CUDA) set(/MP);
    "$<$<AND:$<C_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
    # if (compilerID == MSVC && compilerLanguage != CUDA) set(/Zc:__cplusplus);
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/Zc:__cplusplus>"
    # if (compilerID == MSVC && compilerLanguage == CUDA) set(-Xcompiler "/wd 4819 /Zc:__cplusplus");
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/wd4819 /Zc:__cplusplus>"
)
target_compile_features(
    fakelib INTERFACE
    cxx_std_20
)

add_subdirectory(gpu-half-edge)

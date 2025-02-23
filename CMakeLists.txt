cmake_minimum_required(VERSION 3.9)

# Use *_ROOT environment variables for find_package calls
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
    cmake_policy(SET CMP0074 NEW)
endif()

# Let CAS handle the CUDA architecture flags (for now)
# Windows still gives CMP0104 warning if putting it in cuda.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    cmake_policy(SET CMP0104 OLD)
endif()

project(Ginkgo LANGUAGES C CXX VERSION 1.3.0 DESCRIPTION "A numerical linear algebra library targeting many-core architectures")
set(Ginkgo_VERSION_TAG "develop")
set(PROJECT_VERSION_TAG ${Ginkgo_VERSION_TAG})

# Determine which modules can be compiled
include(cmake/hip_path.cmake)
include(cmake/autodetect_executors.cmake)
include(cmake/build_type_helpers.cmake)

# Ginkgo configuration options
option(GINKGO_DEVEL_TOOLS "Add development tools to the build system" OFF)
option(GINKGO_BUILD_TESTS "Generate build files for unit tests" ON)
option(GINKGO_BUILD_EXAMPLES "Build Ginkgo's examples" ON)
option(GINKGO_BUILD_BENCHMARKS "Build Ginkgo's benchmarks" ON)
option(GINKGO_BUILD_REFERENCE "Compile reference CPU kernels" ON)
option(GINKGO_BUILD_OMP "Compile OpenMP kernels for CPU" ${GINKGO_HAS_OMP})
option(GINKGO_BUILD_DPCPP
    "Compile DPC++ kernels for Intel GPUs or other DPC++ enabled hardware" ${GINKGO_HAS_DPCPP})
option(GINKGO_BUILD_CUDA "Compile kernels for NVIDIA GPUs" ${GINKGO_HAS_CUDA})
option(GINKGO_BUILD_HIP "Compile kernels for AMD or NVIDIA GPUs" ${GINKGO_HAS_HIP})
option(GINKGO_BUILD_DOC "Generate documentation" OFF)
option(GINKGO_SKIP_DEPENDENCY_UPDATE
    "Do not update dependencies each time the project is rebuilt" ON)
option(GINKGO_EXPORT_BUILD_DIR
    "Make Ginkgo export its build directory to the CMake package registry."
    OFF)
option(GINKGO_WITH_CLANG_TIDY "Make Ginkgo call `clang-tidy` to find programming issues." OFF)
option(GINKGO_WITH_IWYU "Make Ginkgo call `iwyu` (Include What You Use) to find include issues." OFF)
option(GINKGO_CHECK_CIRCULAR_DEPS
    "Enable compile-time checks detecting circular dependencies between libraries and non-self-sufficient headers."
    OFF)
option(GINKGO_CONFIG_LOG_DETAILED
    "Enable printing of detailed configuration log to screen in addition to the writing of files," OFF)
option(GINKGO_BENCHMARK_ENABLE_TUNING
    "Enable tuning variables in the benchmarks. For specific use cases, manual code changes could be required."
    OFF)
set(GINKGO_VERBOSE_LEVEL "1" CACHE STRING
    "Verbosity level. Put 0 to turn off. 1 activates a few important messages.")
if(MSVC)
    set(GINKGO_COMPILER_FLAGS "" CACHE STRING
        "Set the required CXX compiler flags, mainly used for warnings. Current default is ``")
else()
    set(GINKGO_COMPILER_FLAGS "-Wpedantic" CACHE STRING
        "Set the required CXX compiler flags, mainly used for warnings. Current default is `-Wpedantic`")
endif()
set(GINKGO_CUDA_COMPILER_FLAGS "" CACHE STRING
    "Set the required NVCC compiler flags, mainly used for warnings. Current default is an empty string")
set(GINKGO_CUDA_ARCHITECTURES "Auto" CACHE STRING
    "A list of target NVIDIA GPU achitectures. See README.md for more detail.")
option(GINKGO_CUDA_DEFAULT_HOST_COMPILER "Tell Ginkgo to not automatically set the CUDA host compiler" OFF)
set(GINKGO_HIP_COMPILER_FLAGS "" CACHE STRING
    "Set the required HIP compiler flags. Current default is an empty string.")
set(GINKGO_HIP_NVCC_COMPILER_FLAGS "" CACHE STRING
    "Set the required HIP nvcc compiler flags. Current default is an empty string.")
set(GINKGO_HIP_HCC_COMPILER_FLAGS "" CACHE STRING
    "Set the required HIP HCC compiler flags. Current default is an empty string.")
set(GINKGO_HIP_CLANG_COMPILER_FLAGS "" CACHE STRING
    "Set the required HIP CLANG compiler flags. Current default is an empty string.")
set(GINKGO_HIP_AMDGPU "" CACHE STRING
    "The amdgpu_target(s) variable passed to hipcc. The default is none (auto).")
option(GINKGO_JACOBI_FULL_OPTIMIZATIONS "Use all the optimizations for the CUDA Jacobi algorithm" OFF)
option(BUILD_SHARED_LIBS "Build shared (.so, .dylib, .dll) libraries" ON)
option(GINKGO_BUILD_HWLOC "Build Ginkgo with HWLOC. Default is ON. If a system HWLOC is not found, then we try to build it ourselves. Switch this OFF to disable HWLOC." ON)

set(GINKGO_CIRCULAR_DEPS_FLAGS "-Wl,--no-undefined")

if(GINKGO_BENCHMARK_ENABLE_TUNING)
    # In this state, the tests and examples cannot be compiled without extra
    # complexity/intrusiveness, so we simply disable them.
    set(GINKGO_BUILD_TESTS OFF)
    set(GINKGO_BUILD_EXAMPLES OFF)
endif()

if(BUILD_SHARED_LIBS AND (WIN32 OR CYGWIN) AND (GINKGO_BUILD_TESTS OR GINKGO_BUILD_EXAMPLES OR GINKGO_BUILD_BENCHMARKS))
    # Change shared libraries output only if this build has executable program
    # with shared libraries.
    set(GINKGO_CHANGED_SHARED_LIBRARY TRUE)
    option(GINKGO_CHECK_PATH "Tell Ginkgo to check if the environment variable PATH is available for this build." ON)
    set(GINKGO_WINDOWS_SHARED_LIBRARY_RELPATH "windows_shared_library" CACHE STRING
        "Set Ginkgo's shared library relative path in windows. Current default is `windows_shared_library`. \
        This absolute path ${PROJECT_BINARY_DIR}/GINKGO_WINDOWS_SHARED_LIBRARY_RELPATH must be in the environment variable PATH.")
    set(GINKGO_WINDOWS_SHARED_LIBRARY_PATH ${PROJECT_BINARY_DIR}/${GINKGO_WINDOWS_SHARED_LIBRARY_RELPATH})
else()
    set(GINKGO_CHANGED_SHARED_LIBRARY FALSE)
endif()

if(GINKGO_BUILD_TESTS AND (GINKGO_BUILD_CUDA OR GINKGO_BUILD_OMP OR GINKGO_BUILD_HIP OR GINKGO_BUILD_DPCPP))
    message(STATUS "GINKGO_BUILD_TESTS is ON, enabling GINKGO_BUILD_REFERENCE")
    set(GINKGO_BUILD_REFERENCE ON CACHE BOOL "Compile reference CPU kernels" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

if(BUILD_SHARED_LIBS)
    set(GINKGO_STATIC_OR_SHARED SHARED)
else()
    set(GINKGO_STATIC_OR_SHARED STATIC)
endif()

# Ensure we have a debug postfix
if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
endif()

if(GINKGO_BUILD_TESTS)
    # Configure CTest
    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/cmake/CTestCustom.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake @ONLY)

    # For testing, we need some special matrices
    add_subdirectory(matrices)

    enable_testing()
    include(CTest)
endif()

if(GINKGO_WITH_CLANG_TIDY)
    find_program(GINKGO_CLANG_TIDY_PATH clang-tidy)
endif()

if(GINKGO_WITH_IWYU)
    find_program(GINKGO_IWYU_PATH iwyu)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules/")

# Find important header files, store the definitions in
# include/ginkgo/config.h.in For details, see
# https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/How-To-Write-Platform-Checks
include(CheckIncludeFileCXX)
check_include_file_cxx(cxxabi.h GKO_HAVE_CXXABI_H)

# Automatically find PAPI and search for the required 'sde' component
set(GINKGO_HAVE_PAPI_SDE 0)
find_package(PAPI OPTIONAL_COMPONENTS sde)
if(PAPI_sde_FOUND)
    set(GINKGO_HAVE_PAPI_SDE 1)
endif()

if(GINKGO_BUILD_HWLOC)
    # By default always use external HWLOC
    set(GINKGO_USE_EXTERNAL_HWLOC 1)
    set(GINKGO_HAVE_HWLOC 1)
else()
    set(GINKGO_USE_EXTERNAL_HWLOC 0)
    set(GINKGO_HAVE_HWLOC 0)
    message(STATUS "HWLOC is being forcibly switched off")
endif()
# Switch off HWLOC for Windows and MacOS
if(MSVC OR WIN32 OR CYGWIN OR APPLE)
    set(GINKGO_HAVE_HWLOC 0)
endif()

set(GINKGO_HIP_PLATFORM_NVCC 0)
set(GINKGO_HIP_PLATFORM_HCC 0)

if(GINKGO_BUILD_HIP)
    # GINKGO_HIPCONFIG_PATH and HIP_PATH are set in cmake/hip_path.cmake
    if(DEFINED ENV{HIP_PLATFORM})
        set(GINKGO_HIP_PLATFORM "$ENV{HIP_PLATFORM}")
    elseif(GINKGO_HIPCONFIG_PATH)
        execute_process(COMMAND ${GINKGO_HIPCONFIG_PATH}
            --platform OUTPUT_VARIABLE GINKGO_HIP_PLATFORM)
    else()
        message(FATAL_ERROR "No platform could be found for HIP. "
            "Set and export the environment variable HIP_PLATFORM.")
    endif()
    message(STATUS "HIP platform set to ${GINKGO_HIP_PLATFORM}")

    if (GINKGO_HIP_PLATFORM STREQUAL "hcc")
        set(GINKGO_HIP_PLATFORM_HCC 1)
    elseif (GINKGO_HIP_PLATFORM STREQUAL "nvcc")
        set(GINKGO_HIP_PLATFORM_NVCC 1)
    endif()
endif()

configure_file(${Ginkgo_SOURCE_DIR}/include/ginkgo/config.hpp.in
    ${Ginkgo_BINARY_DIR}/include/ginkgo/config.hpp @ONLY)

# Load CMake helpers
include(cmake/build_helpers.cmake)
include(cmake/hip_helpers.cmake)
include(cmake/install_helpers.cmake)
include(cmake/windows_helpers.cmake)

# This is modified from
# https://gitlab.kitware.com/cmake/community/wikis/FAQ#dynamic-replace
if(MSVC)
    if(BUILD_SHARED_LIBS)
        ginkgo_switch_to_windows_dynamic("CXX")
        ginkgo_switch_to_windows_dynamic("C")
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    else()
        ginkgo_switch_to_windows_static("CXX")
        ginkgo_switch_to_windows_static("C")
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS FALSE)
    endif()
endif()

# Try to find the third party packages before using our subdirectories
include(cmake/package_helpers.cmake)
ginkgo_find_package(GTest "GTest::GTest;GTest::Main" FALSE 1.8.1)
ginkgo_find_package(gflags gflags FALSE 2.2.2)
ginkgo_find_package(RapidJSON rapidjson TRUE 1.1.0)
if(GINKGO_HAVE_HWLOC)
    ginkgo_find_package(HWLOC hwloc FALSE 2.1)
endif()
set(GKO_HWLOC_XMLFILE "$ENV{HWLOC_XMLFILE}")
message(STATUS "HWLOC XML file set to: ${GKO_HWLOC_XMLFILE}")
add_subdirectory(third_party)    # Third-party tools and libraries

# Ginkgo core libraries
# Needs to be first in order for `CMAKE_CUDA_DEVICE_LINK_EXECUTABLE` to be
# propagated to the other parts of Ginkgo in case of building as static libraries
if(GINKGO_BUILD_CUDA)
    add_subdirectory(cuda)       # High-performance kernels for NVIDIA GPUs
endif()
add_subdirectory(core)           # Core Ginkgo types and top-level functions
add_subdirectory(include)        # Public API self-contained check
if (GINKGO_BUILD_REFERENCE)
    add_subdirectory(reference)  # Reference kernel implementations
endif()
if (GINKGO_BUILD_OMP)
    add_subdirectory(omp)        # High-performance omp kernels
endif()
if (GINKGO_BUILD_DPCPP)
    add_subdirectory(dpcpp)        # High-performance DPC++ kernels
endif()
# HIP needs to be last because it builds the GINKGO_RPATH_FOR_HIP variable
# which needs to know the `ginkgo` target.
if(GINKGO_BUILD_HIP)
    add_subdirectory(hip)        # High-performance kernels for AMD or NVIDIA GPUs
endif()

# Non core directories and targets
if(GINKGO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(GINKGO_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

if(GINKGO_DEVEL_TOOLS)
    add_custom_target(add_license
        COMMAND ${Ginkgo_SOURCE_DIR}/dev_tools/scripts/add_license.sh
        WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR})
    add_dependencies(format add_license)
endif()

# MacOS needs to install bash, gnu-sed, findutils and coreutils
# format_header needs clang-format 6.0.0+
find_program(BASH bash)
if(NOT "${BASH}" STREQUAL "BASH-NOTFOUND" AND GINKGO_DEVEL_TOOLS)
    add_custom_target(generate_ginkgo_header ALL
        COMMAND ${Ginkgo_SOURCE_DIR}/dev_tools/scripts/update_ginkgo_header.sh
        WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR})
    find_program(GIT git)
    if(NOT "${GIT}" STREQUAL "GIT-NOTFOUND")
        add_custom_target(format_header
            COMMAND echo "format header on the modified code files except build/examples/third_party/ginkgo.hpp"
            COMMAND bash -c "git diff --name-only origin/develop...HEAD | \
                grep -Ev 'build|examples|third_party|ginkgo.hpp' | \
                grep -E '(\.hip)?\.(cu|hpp|cuh|cpp)$' | \
                xargs -r -n1 ${Ginkgo_SOURCE_DIR}/dev_tools/scripts/format_header.sh"
            WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR}
            VERBATIM)
    endif()
    unset(GIT CACHE)
    add_custom_target(format_header_all
        COMMAND echo "format header on all code files except build/examples/third_party/ginkgo.hpp"
        COMMAND bash -c "find * -type f | \
                grep -Ev 'build|examples|third_party|ginkgo.hpp' | \
                grep -E '(\.hip)?\.(cu|hpp|cuh|cpp)$' | \
                xargs -r -n1 ${Ginkgo_SOURCE_DIR}/dev_tools/scripts/format_header.sh"
        WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR}
        VERBATIM)
endif()
unset(BASH CACHE)


# Installation
include(cmake/information_helpers.cmake)
ginkgo_interface_information()
ginkgo_git_information()

include(cmake/get_info.cmake)

if(GINKGO_BUILD_DOC)
    add_subdirectory(doc)
endif()

configure_file(${Ginkgo_SOURCE_DIR}/cmake/ginkgo.pc.in
    ${Ginkgo_BINARY_DIR}/ginkgo.pc @ONLY)

# WINDOWS NVCC has " inside the string, add escape character
# to avoid config problem.
ginkgo_modify_flags(CMAKE_CUDA_FLAGS)
ginkgo_modify_flags(CMAKE_CUDA_FLAGS_DEBUG)
ginkgo_modify_flags(CMAKE_CUDA_FLAGS_RELEASE)
ginkgo_install()

if(MSVC)
    # Set path/command with $<CONFIG>
    set(GINKGO_TEST_INSTALL_COMMAND "${Ginkgo_BINARY_DIR}/test_install/$<CONFIG>/test_install")
    set(GINKGO_TEST_EXPORTBUILD_COMMAND "${Ginkgo_BINARY_DIR}/test_exportbuild/$<CONFIG>/test_exportbuild")
    if(GINKGO_BUILD_CUDA)
        set(GINKGO_TEST_INSTALL_COMMAND "${GINKGO_TEST_INSTALL_COMMAND}" "${Ginkgo_BINARY_DIR}/test_install/$<CONFIG>/test_install_cuda")
    endif()
else()
    set(GINKGO_TEST_INSTALL_COMMAND "${Ginkgo_BINARY_DIR}/test_install/test_install")
    set(GINKGO_TEST_EXPORTBUILD_COMMAND "${Ginkgo_BINARY_DIR}/test_exportbuild/test_exportbuild")
    if(GINKGO_BUILD_CUDA)
        set(GINKGO_TEST_INSTALL_COMMAND "${GINKGO_TEST_INSTALL_COMMAND}" "${Ginkgo_BINARY_DIR}/test_install/test_install_cuda")
    endif()
endif()
add_custom_target(test_install
    COMMAND ${CMAKE_COMMAND} -G${CMAKE_GENERATOR} -H${Ginkgo_SOURCE_DIR}/test_install
    -B${Ginkgo_BINARY_DIR}/test_install
    -DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_CONFIG_DIR}
    -DHWLOC_DIR=${HWLOC_DIR}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
    # `--config cfg` is ignored by single-configuration generator.
    # `$<CONFIG>` is always be the same as `CMAKE_BUILD_TYPE` in
    # single-configuration generator.
    COMMAND ${CMAKE_COMMAND} --build ${Ginkgo_BINARY_DIR}/test_install --config $<CONFIG>
    COMMAND ${GINKGO_TEST_INSTALL_COMMAND}
    COMMENT "Running a test on the installed binaries. This requires running `(sudo) make install` first.")

add_custom_target(test_exportbuild
    COMMAND ${CMAKE_COMMAND} -G${CMAKE_GENERATOR} -H${Ginkgo_SOURCE_DIR}/test_exportbuild
    -B${Ginkgo_BINARY_DIR}/test_exportbuild
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
    # `--config cfg` is ignored by single-configuration generator.
    # `$<CONFIG>` is always be the same as `CMAKE_BUILD_TYPE` in
    # single-configuration generator.
    COMMAND ${CMAKE_COMMAND} --build ${Ginkgo_BINARY_DIR}/test_exportbuild --config $<CONFIG>
    COMMAND ${GINKGO_TEST_EXPORTBUILD_COMMAND}
    COMMENT "Running a test on Ginkgo's exported build directory. "
    "This requires compiling Ginkgo with `-DGINKGO_EXPORT_BUILD_DIR=ON` first.")


# Setup CPack
set(CPACK_PACKAGE_DESCRIPTION_FILE "${Ginkgo_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${Ginkgo_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_ICON "${Ginkgo_SOURCE_DIR}/assets/logo.png")
set(CPACK_PACKAGE_CONTACT "ginkgo.library@gmail.com")
include(CPack)

# And finally, print the configuration to screen:
if(GINKGO_CONFIG_LOG_DETAILED)
    FILE(READ ${PROJECT_BINARY_DIR}/detailed.log GINKGO_LOG_SUMMARY)
else()
    FILE(READ ${PROJECT_BINARY_DIR}/minimal.log GINKGO_LOG_SUMMARY)
endif()
MESSAGE("${GINKGO_LOG_SUMMARY}")

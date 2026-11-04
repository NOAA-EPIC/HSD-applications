cmake_minimum_required(VERSION 3.22)
project(KokkosProtos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Backend Options
# ---------------------------------------------------------------------------
option(Kokkos_ENABLE_SERIAL "Enable Kokkos Serial backend" ON)
option(Kokkos_ENABLE_OPENMP "Enable Kokkos OpenMP backend" ON)
option(Kokkos_ENABLE_CUDA   "Enable Kokkos CUDA backend"   OFF)

if(Kokkos_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # Default to sm_75 (Tesla T4 on GitHub-hosted larger runner)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    set(Kokkos_ARCH_TURING75 ON CACHE BOOL "Enable Turing75 for Kokkos" FORCE)
endif()

if(Kokkos_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Fetch Kokkos via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        4.3.01
)
FetchContent_MakeAvailable(kokkos)

# ---------------------------------------------------------------------------
# Target Executable
# ---------------------------------------------------------------------------
add_executable(kokkos_runner src/main.cpp)
target_link_libraries(kokkos_runner PRIVATE Kokkos::kokkos)

if(Kokkos_ENABLE_CUDA)
    # Compile main.cpp with nvcc when CUDA backend is active
    set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)
endif()

if(Kokkos_ENABLE_OPENMP)
    target_link_libraries(kokkos_runner PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# CTest Harness
# ---------------------------------------------------------------------------
enable_testing()

# Serial Backend Test
add_test(NAME test_serial COMMAND kokkos_runner --backend=serial)
set_tests_properties(test_serial PROPERTIES LABELS "cpu;serial;ci")

# OpenMP Backend Test
if(Kokkos_ENABLE_OPENMP)
    add_test(NAME test_openmp COMMAND kokkos_runner --backend=openmp)
    set_tests_properties(test_openmp PROPERTIES
        LABELS "cpu;openmp;ci"
        ENVIRONMENT "OMP_NUM_THREADS=4"
    )
endif()

# CUDA Backend Test
if(Kokkos_ENABLE_CUDA)
    add_test(NAME test_cuda COMMAND kokkos_runner --backend=cuda)
    set_tests_properties(test_cuda PROPERTIES LABELS "gpu;cuda;ci")
endif()

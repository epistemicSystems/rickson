cmake_minimum_required(VERSION 3.18)
project(zs_evm_cuda LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture (adjust based on target GPU)
# RTX 2060+: sm_75, RTX 3000: sm_86, RTX 4000: sm_89
set(CMAKE_CUDA_ARCHITECTURES "75;86;89")

# Source files
set(CUDA_SOURCES
    kernels/pyramid.cu
    kernels/temporal_filter.cu
)

# Create shared library
add_library(zs_evm_cuda SHARED ${CUDA_SOURCES})

# Set properties
set_target_properties(zs_evm_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(zs_evm_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA libraries
target_link_libraries(zs_evm_cuda
    CUDA::cudart
    CUDA::cuda_driver
)

# Compiler options
target_compile_options(zs_evm_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --ptxas-options=-v
        -lineinfo
    >
)

# Install
install(TARGETS zs_evm_cuda
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

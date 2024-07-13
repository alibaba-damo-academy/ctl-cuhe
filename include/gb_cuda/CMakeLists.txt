add_library(gb_cuda
init.cu
Bootstrap.cu
Pack.cu
gb_cuda.cpp
)
target_compile_features(gb_cuda PRIVATE cxx_std_14)

target_include_directories(gb_cuda
    PRIVATE ${PROJECT_SOURCE_DIR}/include/gb_cuda)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -cudart static -use_fast_math -lcudadevrt -O2 -Xptxas -v -maxrregcount 64 -lineinfo -g")

set_target_properties(gb_cuda
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(gb_cuda PUBLIC SEAL::seal)
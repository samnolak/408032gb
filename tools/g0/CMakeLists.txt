# Out-of-tree build of the Gate 0 tool against a pinned llama.cpp checkout (no upstream edits).
#   cmake -S tools/g0 -B build-g0 -DLLAMA_CPP_DIR=/path/to/llama.cpp -DGGML_CUDA=ON \
#         -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-g0 -j --target g0-expert-hist
cmake_minimum_required(VERSION 3.18)
project(g0_expert_hist C CXX)

set(LLAMA_CPP_DIR "" CACHE PATH "llama.cpp checkout at ec7630a640789c393694fb194f1bbbf0369fc62d")
if (NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "set -DLLAMA_CPP_DIR to a llama.cpp checkout")
endif()

set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_CPP_DIR} llama.cpp)

add_executable(g0-expert-hist expert_hist.cpp)
target_link_libraries(g0-expert-hist PRIVATE llama-common llama)
target_compile_features(g0-expert-hist PRIVATE cxx_std_17)

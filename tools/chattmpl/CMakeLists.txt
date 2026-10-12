# Out-of-tree build of the N14 chat parse harness against a pinned llama.cpp checkout (no upstream edits).
#   cmake -S tools/chattmpl -B build-chattmpl -DLLAMA_CPP_DIR=/path/to/llama.cpp -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-chattmpl -j --target chat-parse-harness
# tools/chattmpl/run_n14.sh links the same source against an existing shared build instead (faster).
cmake_minimum_required(VERSION 3.18)
project(chattmpl CXX)

set(LLAMA_CPP_DIR "" CACHE PATH "llama.cpp checkout at 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 (R1 pin)")
if (NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "set -DLLAMA_CPP_DIR to a llama.cpp checkout")
endif()

set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_CPP_DIR} llama.cpp)

add_executable(chat-parse-harness chat_parse_harness.cpp)
target_link_libraries(chat-parse-harness PRIVATE llama-common llama)
target_compile_features(chat-parse-harness PRIVATE cxx_std_17)

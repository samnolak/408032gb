# Out-of-tree CPU build of ik_llama.cpp at the pin of tools/engines/pins.json (5194a9e1451b885e24dd5a065f650864e8807c59)
# plus our tensor dump tool for it (tools/dump/ik_dump_tensors.cpp); no upstream edits. N12, gap 5 of
# docs/reassessment/CORRECTNESS.md. Used by tools/tinygen/run_parity_ik_glm5next.sh:
#   cmake -S tools/ik -B build-ik -DIK_LLAMA_DIR=/path/to/ik_llama.cpp -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
#         -DLLAMA_CURL=OFF
#   cmake --build build-ik -j --target ik-dump-tensors        (output: build-ik/ik-dump-tensors)
# ik's llama-server cannot be built this way: its sources include <src/llama-impl.h> from CMAKE_SOURCE_DIR
# (examples/server/CMakeLists.txt:80, server-common.h:6 @5194a9e), which is this directory here; build the server from
# the ik checkout itself (cmake -S <ik> -B <build> with the same flags), as scripts/engines/build_engine.py does.
# The ik flags are those of pins.json without the four CUDA ones; GGML_NATIVE=OFF gives AVX2 without AVX-512 at that
# commit (ggml/CMakeLists.txt:75-92: GGML_AVX/AVX2/FMA/F16C default ON, every AVX512 option OFF).
cmake_minimum_required(VERSION 3.18)
project(ik_tools C CXX)

set(IK_LLAMA_DIR "" CACHE PATH "ik_llama.cpp checkout at 5194a9e1451b885e24dd5a065f650864e8807c59")
if (NOT EXISTS "${IK_LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "set -DIK_LLAMA_DIR to an ik_llama.cpp checkout")
endif()

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)   # common/ is added unconditionally (CMakeLists.txt:232 @5194a9e)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
add_subdirectory(${IK_LLAMA_DIR} ik_llama.cpp)

add_executable(ik-dump-tensors ../dump/ik_dump_tensors.cpp)
target_link_libraries(ik-dump-tensors PRIVATE common llama)
target_compile_features(ik-dump-tensors PRIVATE cxx_std_17)

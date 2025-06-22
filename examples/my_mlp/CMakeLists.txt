#
# mlp-backend

set(TEST_TARGET mlp-backend)
add_executable(${TEST_TARGET} mlp-backend.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

# Add backend support
if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()

# Optional: Add any additional dependencies if needed
# target_link_libraries(${TEST_TARGET} PRIVATE ggml some_other_lib) 
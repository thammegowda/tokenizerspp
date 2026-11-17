function(tokenizerpp_optimize_bench target)
    if(APPLE)
        target_link_options(${target} PRIVATE -Wl,-dead_strip -Wl,-no_exported_symbols)
    elseif(UNIX)
        target_link_options(${target} PRIVATE -Wl,--gc-sections -s)
    endif()
    if(APPLE AND CMAKE_STRIP)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${CMAKE_STRIP}" "$<TARGET_FILE:${target}>"
            VERBATIM
        )
    endif()
endfunction()

add_executable(bench_cpp bench_cpp.cpp)
target_link_libraries(bench_cpp PRIVATE tokenizerpp)
set_target_properties(bench_cpp PROPERTIES
    OUTPUT_NAME "bench_cpp.out"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
tokenizerpp_optimize_bench(bench_cpp)

add_executable(bench_regex bench_regex.cpp)
target_link_libraries(bench_regex PRIVATE tokenizerpp)
set_target_properties(bench_regex PROPERTIES
    OUTPUT_NAME "bench_regex.out"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
tokenizerpp_optimize_bench(bench_regex)

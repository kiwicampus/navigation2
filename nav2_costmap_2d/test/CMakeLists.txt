set(TEST_MAP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/map)
set(TEST_LAUNCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test_launch_files)

add_subdirectory(unit)
add_subdirectory(integration)
add_subdirectory(regression)
# Benchmarks call google benchmark 1.7 APIs (State::thread_index()); Jammy ships 1.6.
# add_subdirectory(benchmark)

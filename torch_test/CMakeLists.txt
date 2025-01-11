cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)


find_package(Torch REQUIRED)

add_executable(torch_test src/torch_script.cpp)
target_link_libraries(torch_test "${TORCH_LIBRARIES}")
set_property(TARGET torch_test PROPERTY CXX_STANDARD 17)
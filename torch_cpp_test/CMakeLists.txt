cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-g)
project(temp_test)

# Torch 
find_package(Torch REQUIRED)

# Find PortAudio
find_library(PORTAUDIO_LIB portaudio REQUIRED PATHS /usr/local/lib /usr/lib)
find_path(PORTAUDIO_INCLUDE_DIR portaudio.h REQUIRED PATHS /usr/local/include /usr/include)

# Verify if found
if(NOT PORTAUDIO_LIB OR NOT PORTAUDIO_INCLUDE_DIR)
    message(FATAL_ERROR "PortAudio not found!")
endif()

message("PortAudio Library: ${PORTAUDIO_LIB}")
message("PortAudio Include: ${PORTAUDIO_INCLUDE_DIR}")

# Create Exe 1
add_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/src/greedy_decoder.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/decoders.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}")
target_include_directories(test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create Exe 2
add_executable(test_handler ${CMAKE_CURRENT_SOURCE_DIR}/src/exp.cpp 
               ${CMAKE_CURRENT_SOURCE_DIR}/src/decoders.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/src/my_utils.cpp
                )
target_link_libraries(test_handler ${TORCH_LIBRARIES} ${PORTAUDIO_LIB})
target_include_directories(test_handler PUBLIC ${PORTAUDIO_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Create Exe 3 (This is temproary)
add_executable(nbl ${CMAKE_CURRENT_SOURCE_DIR}/src/nbl.cpp)
target_link_libraries(nbl ${TORCH_LIBRARIES})


set_property(TARGET test PROPERTY CXX_STANDARD 17)
set_property(TARGET test_handler PROPERTY CXX_STANDARD 17)



message("Torch library at ${TORCH_LIBRARIES}")